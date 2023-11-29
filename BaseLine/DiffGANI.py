import numpy as np
import pandas as pd
import torch
import random
import Mean
from get_attr_attn import get_attr_map_Null_kendall
from model import Diffusion, Discriminator_model
from model.FD_model import get_FD_model_Tree
from util import sort_corr, init_attn_2, categorical_to_code, Data_convert, get_M_by_data_m, get_number_data_mu_var

def softmax(series):
    exps = np.exp(series - np.max(series))
    return exps / exps.sum()
def get_Diff_acc_RMSE(nan_data, path, miss_rate, miss_data, enc, data_m, categorical_cols, continuous_cols,value_cat, device, param, label_data,values,ori_data,label_num, args):
    pathMiss = path + 'miss_data_{}.csv'.format(miss_rate)
    corr_map = get_attr_map_Null_kendall(pathMiss, data_m, categorical_cols, continuous_cols)
    sort_corr_dict = sort_corr(corr_map)
    attention_data = Mean.fill_data_mean(nan_data, continuous_cols, categorical_cols)
    attention_impute_data = pd.DataFrame(attention_data,columns=values)
    if args.UseAttention == 'True':
        attention_impute_data, impute_code = init_attn_2(corr_map, miss_data, data_m, categorical_cols, enc,
                                                                    value_cat, device, param['top_k'])
        attention_impute_data.to_csv(path + 'Attention_Input_miss_data_{}.csv'.format(miss_rate), index=None)
        attention_impute_data = pd.read_csv(path + 'Attention_Input_miss_data_{}.csv'.format(miss_rate))
    data_num = len(attention_impute_data)
    d_input_dim = attention_impute_data.shape[1]
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
    if args.UseFD == "True":
        FD_model_list = get_FD_model_Tree(miss_data, data_m, categorical_cols, zero_feed_data_code, fields, device, sort_corr_dict)
    else:
        FD_model_list = []
    input_dim = feed_data.shape[1]
    encoder_dim = input_dim
    encoder_out_dim = input_dim
    decoder_dim = random.choice([500, 400, 300, 200, 100])
    latent_dim = input_dim // 2
    torch.manual_seed(3407)
    num_steps = param['T']
    diffusion = Diffusion.Diffusion(input_dim, input_dim, input_dim, num_steps + 1)
    ema = Diffusion.EMA(decay=0.99)
    ema.register(diffusion)
    Discriminator = Discriminator_model.D(input_dim, latent_dim, d_input_dim)
    Discriminator_denoise_x = Discriminator_model.Discriminator_noise_x(input_dim * 2, latent_dim, d_input_dim,num_steps + 1)
    Generator_x0 = Diffusion.Generator_x0(input_dim, input_dim * 2, input_dim, num_steps + 1, fields)
    ARMSE,AMAE = Diffusion.train_diffusion_discriminator(Discriminator, Generator_x0, Discriminator_denoise_x, FD_model_list,
                                            num_steps, param["epochs"], param["lr"], param['batch_size'], param["loss_weight"], data_m,
                                            impute_data_code, label_data, fields, value_cat, values,
                                            attention_impute_data, enc, ori_data, continuous_cols, label_num, device,
                                            args.UseLearner)
    return ARMSE,AMAE