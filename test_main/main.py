# -*- coding: utf-8 -*-
import statistics

import math
import numpy as np
import pandas as pd
import torch
import argparse
import json
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

# import DiffGANI
# import VAIM
# import GAIN
# import MissForest
# from BaseLine.hyperimpute_main import test
# from BaseLine.not_MIWAE import fill_NOT
import util
# from hyperimpute_main import test
# from model.Learner import train_L_code, L
from param.data_index import get_data_index
import os
from utils.data_loader import data_loader, value_loader, set_label
from utils.util import categorical_to_code
from BaseLine import DiffGANI,Mean
# import torchvision


def get_run_index(j):
    if j == 0:
        return 0.1, "MCAR"
    elif j == 1:
        return 0.2, "MCAR"
    elif j == 2:
        return 0.3, "MCAR"
    elif j == 3:
        return 0.4, "MCAR"
    elif j == 4:
        return 0.5, "MCAR"
    elif j == 5:
        return 0.2, "MAR"
    elif j == 6:
        return 0.2, "MNAR"
    elif j == 7:
        return 0.2, "Region"

def get_data_name(j):
    if j == 0:
        return "wine"
    elif j == 1:
        return "wireless"
    elif j == 2:
        return "spam"
    elif j == 3:
        return "adult"
    elif j == 4:
        return "hospital"
    elif j == 5:
        return "tax"


if __name__ == "__main__":

    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='a json config file', default='param/param.json')
    parser.add_argument('--Data', type=str, help='data_name', default='wine')
    parser.add_argument('--MissType', type=str,help='miss_type', default='MCAR')
    parser.add_argument('--MissRate', type=float, help='miss_rate', default='0.2')
    parser.add_argument('--AllSeed', type=int, help='miss_seed', default=42)
    parser.add_argument('--UseAttention', type=str, help='use or not use Attention', default='True')
    parser.add_argument('--UseLearner', type=str, help='use or not use Data-Utils plug-in', default='True')
    parser.add_argument('--UseFD', type=str, help='use or not use Data-Dependency plug-in', default='True')

    args = parser.parse_args()

    # args.MissRate, args.MissType = get_run_index(r)
    # args.Data = get_data_name(r)

    with open(args.config) as f:
        params = json.load(f)
    try:
        os.mkdir("../exp-dir")
    except FileExistsError:
        pass
    dataset_name = args.Data
    use_A = args.UseAttention
    use_L = args.UseLearner
    use_FD = args.UseFD
    data_index = get_data_index(dataset_name, params)
    if data_index == -1:
        pass
    param = params[data_index]
    path = "exp-dir/"+param["name"]+"/"
    try:
        os.mkdir("exp-dir/"+param["name"])
    except FileExistsError:
        pass
    data_name = param["name"]
    loss_weight = param["loss_weight"]
    ground_truth_file = param["file_path"]
    label = "True"
    label_file_path = path
    label_file, label_num = set_label(ground_truth_file, label_file_path, label)
    categorical_cols = param['categorical_cols']
    top_k = param['top_k']
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    all_ARMSE, all_AMAE, all_ACC = 0, 0, 0
    ARMSE_list, AMAE_list, ACC_list, fd_list = [],[],[],[]

    # run 5 times
    for k in range(5):
        miss_rate, miss_type, seed = float(args.MissRate), args.MissType, args.AllSeed
        # ori_data: The original, clean training dataset.
        # miss_data: The dataset with missing values filled with 0 or 'NULL'.
        # data_m: Represents the missing status, where '0' indicates a missing value.
        # label_data: The label column of the training set.
        # true_data: Rows in the dataset that do not have any missing data.
        # continuous_cols: Columns in the dataset that contain numeric data.
        # nan_data: The dataset where missing values are filled with NaN.
        # test_data: Data used for testing downstream tasks.
        # all_data: test_data + ori_data for enc
        ori_data, miss_data, data_m, label_data, true_data, true_label, continuous_cols, nan_data, test_data, all_data = data_loader(label_file, miss_rate, categorical_cols, label, miss_type, seed)
        values, value_num, value_cat = value_loader(label_file, continuous_cols, label)
        miss_data.columns = values
        miss_data.to_csv(path + 'miss_data_{}.csv'.format(miss_rate), index=None)
        # We learn the encoding rules solely from the raw data and do not use them subsequently.
        ori_code, enc = categorical_to_code(all_data, value_cat, enc=None)


        # Diffusion_GAN
        fill_data_np = DiffGANI.get_Diff_acc_RMSE(nan_data, path, miss_rate, miss_data, enc, data_m, categorical_cols, continuous_cols,value_cat, device, param, label_data,values,ori_data,label_num,args)

        # Mean
        # fill_data_np = Mean.fill_data_mean(nan_data, continuous_cols)

        # MissFI
        # fill_data_np = MissForest.fill_data_missForest(data_m, nan_data, continuous_cols, categorical_cols)

        # KNNI
        # fill_data_mean_np = KNNI.sim_impute(nan_data, continuous_cols, categorical_cols, data_m)

        # GAIN
        # fill_data_np = GAIN.get_GAIN_filled(data_m, nan_data, continuous_cols, categorical_cols,value_cat, enc, values, device)

        # NOT-MIWAE
        # fill_data_np = fill_NOT.get_notMIWAE_filled(data_m, nan_data, continuous_cols, categorical_cols, value_cat, enc, values, device)

        # Hyper
        # fill_data_np = test.fill_data_hyper(nan_data)

        # VAIM
        # fill_data_np = VAIM.get_VAIM_filled(data_m, nan_data, continuous_cols, categorical_cols,value_cat, enc, values, device, param, label_data,ori_data, label_num)

        # Calculate ARMSE, AMAE
        fill_data = pd.DataFrame(fill_data_np,columns=values)
        # cat_to_code_data, enc = categorical_to_code(ori_data.copy(), value_cat, enc)
        ARMSE, AMAE = util.errorLoss(fill_data, ori_data, data_m, value_cat, continuous_cols, enc)

        # Calculate ACC
        Accuracy = util.get_down_acc(fill_data, label_data, test_data, value_cat, continuous_cols, enc, seed)
        # if k == 0:
        #     continue
        print("数据集为：{}, ARMSE为：{:.4f}, AMAE为:{:.4f}, 下游任务准确率为：{:.4f}".format(data_name, ARMSE, AMAE, Accuracy))

        #EGG-GAE : run run.py. We design consistent random seeds to ensure consistency of missing data
        ARMSE_list.append(ARMSE)
        AMAE_list.append(AMAE)
        ACC_list.append(Accuracy)

        all_ARMSE = all_ARMSE + ARMSE
        all_AMAE = all_AMAE + AMAE
        all_ACC = all_ACC + Accuracy
    ARMSE, AMAE, Accuracy = all_ARMSE / 5., all_AMAE/5., all_ACC/5.
    ARMSE_std, AMAE_std, ACC_std= statistics.pstdev(ARMSE_list), statistics.pstdev(AMAE_list), statistics.pstdev(ACC_list)
    print("dataset：{}， missrate：{}， misstype：{}，ARMSE：{:.4f}±{:.4f}，AMAE：{:.4f}±{:.4f}, ACC: {:.4f}±{:.4f}, Use_A: {}, Use_L:{}, Use_FD:{}".format(
                data_name, miss_rate, miss_type, ARMSE, ARMSE_std, AMAE, AMAE_std, Accuracy, ACC_std,use_A, use_L,use_FD))



