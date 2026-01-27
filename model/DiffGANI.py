import numpy as np
import pandas as pd
import torch
import random
from BaseLine import Mean
from get_attr_attn import get_attr_map_Null_kendall
from model import Diffusion, Discriminator_model
from model.FD_model import get_FD_model_Tree, get_FD_model_Tree_GroundTruth, get_FD_model_Tree_GroundTruth_From_Complete_Data, set_fd_tau, get_CFD_model_Tree, get_CFD_GroundTruth_model_Tree
from util import sort_corr, init_attn_2, categorical_to_code, Data_convert, get_M_by_data_m, get_number_data_mu_var

def build_col_idx_to_unique_values(data, categorical_cols):
    """
    Build a mapping from column index to unique values array for decoding encoded values.
    This uses np.unique which returns sorted unique values.
    """
    col_idx_to_unique_values = {}
    data_values = data.values if hasattr(data, 'values') else data
    for col_idx, col_val in enumerate(data_values.T):
        if col_idx in categorical_cols:
            arr = np.array(col_val).astype(str)
            unique_values, _ = np.unique(arr, return_inverse=True)
            col_idx_to_unique_values[col_idx] = unique_values
    return col_idx_to_unique_values

def softmax(series):
    exps = np.exp(series - np.max(series))
    return exps / exps.sum()
def get_Diff_acc_RMSE(nan_data, path, miss_rate, miss_data, enc, data_m, categorical_cols, continuous_cols,value_cat, device, param, label_data,values,ori_data,label_num, args):
    pathMiss = path + 'miss_data_{}.csv'.format(miss_rate)
    corr_map = get_attr_map_Null_kendall(pathMiss, data_m, categorical_cols, continuous_cols)

    sort_corr_dict = sort_corr(corr_map)

    attention_data = Mean.fill_data_mean(nan_data, continuous_cols)
    attention_impute_data = pd.DataFrame(attention_data,columns=values)
    if args.UseAttention:
        print('Begin Attention')
        attention_impute_data, impute_code = init_attn_2(corr_map,  attention_impute_data, data_m, categorical_cols, enc,
                                                                    value_cat, device, param['top_k'])
        attention_impute_data.to_csv(path + 'Attention_Input_miss_data_{}.csv'.format(miss_rate), index=None)
        attention_impute_data = pd.read_csv(path + 'Attention_Input_miss_data_{}.csv'.format(miss_rate))
        print('End Attention')
    data_num = len(attention_impute_data)
    d_input_dim = attention_impute_data.shape[1]
    attention_impute_data[value_cat] = attention_impute_data[value_cat].astype(str)
    cat_to_code_data, enc = categorical_to_code(attention_impute_data.copy(), value_cat, enc)
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    fields, feed_data = Data_convert(cat_to_code_data, param['model_name'], continuous_cols)
    M_tensor = get_M_by_data_m(data_m, fields, device)
    impute_data_code = torch.tensor(feed_data.values, dtype=torch.float).to(device)
    zero_feed_data_code = impute_data_code * M_tensor
    zero_feed_data = pd.DataFrame(np.array(zero_feed_data_code.cpu()))
    zero_feed_data.columns = feed_data.columns
    number_data_mu_var = get_number_data_mu_var(zero_feed_data_code, M_tensor, fields, device)
    row_indices = torch.nonzero(torch.all(M_tensor == 1, dim=1)).squeeze().tolist()
    true_data_code = zero_feed_data_code[row_indices]
    # Set FD tau from dataset parameters (default 3)
    try:
        set_fd_tau(param.get('tau', 3))
    except Exception:
        set_fd_tau(3)
        
    # Initialize default values
    FD_model_list = []
    CFD_model_list = []
    dict_A = {}  # key=(LHS, RHS), value=pattern
    dict_B = {}  # key=(LHS, RHS), value=list of (pattern, support) sorted by support descending
    dict_cnt = {}  # key=(LHS, RHS), value=number of CFDs for this structure
    min_rows = param.get('cfd_support_min_rows', 5)
    max_pattern_per_structure = param.get('cfd_max_pattern_per_structure', 5) # Maximum number of different patterns per (LHS, RHS) structure
    use_CFD_refine = False  # Default: do not use CFD refine

    if args.UseFD == 'True':
        print('Begin FD_model')
        FD_model_list = get_FD_model_Tree(miss_data, data_m, categorical_cols, zero_feed_data_code, fields, device, sort_corr_dict,param)
        # FD_model_list = get_FD_model_Tree_GroundTruth_From_Complete_Data(ori_data, categorical_cols, zero_feed_data_code, fields, device, sort_corr_dict, param)
        #                                     sort_corr_dict)
        print('End FD_model')
        # In FD mode, CFD related variables keep default values, set use_CFD_refine to False
        use_CFD_refine = False

    elif args.UseCFD == 'True':
        print('Begin CFD_model')
        # FD_model_list remains empty (default value)
        # Build decoding mapping from original data for consistent decoding
        col_idx_to_unique_values = build_col_idx_to_unique_values(ori_data, categorical_cols)
        CFD_model_list = get_CFD_GroundTruth_model_Tree(ori_data, categorical_cols, zero_feed_data_code, fields, device, sort_corr_dict, param, min_rows, col_idx_to_unique_values=col_idx_to_unique_values)
        CFD_model_list, dict_A, dict_B, dict_cnt = get_CFD_model_Tree(miss_data, data_m, categorical_cols, zero_feed_data_code, fields, device, sort_corr_dict,param,min_rows, col_idx_to_unique_values=col_idx_to_unique_values)
        print('End CFD_model')
        # In CFD mode, set use_CFD_refine to True
        use_CFD_refine = True
    else:
        pass
    
    # return 
    input_dim = feed_data.shape[1]
    encoder_dim = input_dim
    encoder_out_dim = input_dim
    decoder_dim = random.choice([500, 400, 300, 200, 100])
    latent_dim = input_dim // 2
    torch.manual_seed(3407)
    num_steps = param['T']
    diffusion = Diffusion.Diffusion(input_dim, input_dim, input_dim, num_steps + 1)
    # from calc_source.flops import compute_diffusion_flops
    # flops, params = compute_diffusion_flops(diffusion, input_dim, num_steps + 1, device)
    # total_flops = flops * param['steps_per_epoch'] * param['T'] * param['epochs'] * 2
    # with open('calc_source/flops.txt', 'a') as file:
    #     file.write(f"{param['name']}: DDGANI FLOPS: {total_flops}, Params: {params}\n")
    ema = Diffusion.EMA(decay=0.99)
    ema.register(diffusion)
    Discriminator = Discriminator_model.D(input_dim, latent_dim, d_input_dim)
    Discriminator_denoise_x = Discriminator_model.Discriminator_noise_x(input_dim * 2, latent_dim, d_input_dim,num_steps + 1)
    # Generator_x0 = Diffusion.Generator_x0(input_dim, input_dim * 2, input_dim, num_steps + 1, fields)
    Generator_x0 = Diffusion.Generator_x0_Attention(input_dim, num_steps + 1, update_corr_map(fields, corr_map), fields)

    # Pass dataset_name and run_name (here using param['name'] for both, can be replaced if different)
    impute_data, eval_time, best_time, res_dict = Diffusion.train_diffusion_discriminator(Discriminator, Generator_x0, Discriminator_denoise_x, FD_model_list, CFD_model_list, dict_A, dict_B, dict_cnt, min_rows, max_pattern_per_structure, use_CFD_refine,
                                            num_steps, param["epochs"], param["lr"], param['batch_size'], param["loss_weight"], data_m,
                                            impute_data_code, label_data, fields, value_cat, values,
                                            attention_impute_data, enc, ori_data, continuous_cols, label_num, device,
                                            param.get('name', 'dataset'), param.get('name', 'run'), args.UseLearner,
                                            col_idx_to_unique_values)
    return impute_data, eval_time, best_time, res_dict


def update_corr_map(fields, corr_map):
    dimensions = [field.dim() if field.data_type == "Categorical Data" else 1 for field in fields]
    # Calculate the expanded matrix size
    expanded_size = sum(dimensions)
    # Initialize the expanded matrix
    expanded_corr_map = [[0]*expanded_size for _ in range(expanded_size)]
    # Fill in the expanded matrix
    current_row = 0
    for i in range(len(fields)):
        dim = dimensions[i]
        current_col = 0
        for j in range(len(fields)):
            sub_dim = dimensions[j]
            # Fill the i-th row block, j-th column block
            for k in range(dim):
                for l in range(sub_dim):
                    expanded_corr_map[current_row + k][current_col + l] = corr_map[i][j]
            current_col += sub_dim
        current_row += dim
    return expanded_corr_map