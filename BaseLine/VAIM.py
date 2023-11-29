import numpy as np
import pandas as pd
import torch
import random

import KNNI
import Mean
from get_attr_attn import get_attr_map_Null_kendall
from model import Diffusion, Discriminator_model, VAE_model
from model.FD_model import get_FD_model_Tree
from util import sort_corr, init_attn_2, categorical_to_code, Data_convert, get_M_by_data_m, get_number_data_mu_var


def get_VAIM_filled(path, miss_rate, miss_data, data_m, nan_data, continuous_cols, categorical_cols,value_cat, enc, values, device, param, label_data, ori_data, label_num, val_data):

    # pathMiss = path + 'miss_data_{}.csv'.format(miss_rate)
    # corr_map = get_attr_map_Null_kendall(pathMiss, data_m, categorical_cols, continuous_cols)
    # sort_corr_dict = sort_corr(corr_map)
    # attention_impute_data, impute_code = init_attn_2(corr_map, miss_data, data_m, categorical_cols, enc,
    #                                                             value_cat, device, param['model_name'], param['top_k'])
    attention_impute_data = KNNI.sim_impute(nan_data, continuous_cols, categorical_cols, data_m)
    attention_impute_data = pd.DataFrame(attention_impute_data, columns=values)
    data_num = len(attention_impute_data)
    d_input_dim = attention_impute_data.shape[1]
    cat_to_code_data, enc = categorical_to_code(attention_impute_data.copy(), value_cat, enc)
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    fields, feed_data = Data_convert(cat_to_code_data, "minmax", continuous_cols)
    M_tensor = get_M_by_data_m(data_m, fields, device)
    impute_data_code = torch.tensor(feed_data.values, dtype=torch.float).to(device)
    zero_feed_data_code = impute_data_code * M_tensor  # 为0的data
    zero_feed_data = pd.DataFrame(np.array(zero_feed_data_code.cpu()))
    zero_feed_data.columns = feed_data.columns
    number_data_mu_var = get_number_data_mu_var(zero_feed_data_code, M_tensor, fields, device)
    row_indices = torch.nonzero(torch.all(M_tensor == 1, dim=1)).squeeze().tolist()
    true_data_code = zero_feed_data_code[row_indices]
    input_dim = feed_data.shape[1]
    encoder_dim = random.choice([500,400,300,200,100])
    encoder_out_dim = random.choice([500,400,300,200,100])
    decoder_dim = random.choice([500,400,300,200,100])
    latent_dim = random.choice([16,32,64,128])
    torch.manual_seed(3407)
    VAE = VAE_model.Vae(input_dim, encoder_dim, encoder_out_dim, latent_dim, fields)
    Discriminator = Discriminator_model.D(input_dim, latent_dim, d_input_dim)
    ARMSE,AMAE,Acc = VAE_model.VAE_Dis_train(True, VAE, Discriminator, param['epochs'], param['steps_per_epoch'], param['batch_size'], param['loss_weight'], data_m, impute_data_code, label_data, fields, value_cat, values,attention_impute_data,enc,
                            ori_data,continuous_cols,label_num,device,param['name'], val_data)
    return ARMSE,AMAE,Acc