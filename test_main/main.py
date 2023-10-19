# -*- coding: utf-8 -*-
import torch
import argparse
import json
import util
from param.data_index import get_data_index
import os
from utils.data_loader import data_loader, value_loader, set_label
from utils.util import categorical_to_code
from BaseLine import DiffGANI
import torchvision


if __name__ == "__main__":
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torchvision.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='a json config file', default='param/param.json')
    args = parser.parse_args()
    with open(args.config) as f:
        params = json.load(f)
    try:
        os.mkdir("../exp-dir")
    except FileExistsError:
        pass
    dataset_name = ['adult']
    for i in range(0,1):
        data_index = get_data_index(dataset_name[i], params)
        if data_index == -1:
            break
        param = params[data_index]
        path = "exp-dir/"+param["name"]+"/"
        try:
            os.mkdir("exp-dir/"+param["name"])
        except FileExistsError:
            pass
        data_name = param["name"]
        loss_weight = param["loss_weight"]
        ground_truth_file = param["file_path"]
        label = param['use_Learner']
        label_file_path = path
        label_file, label_num = set_label(ground_truth_file, label_file_path, label)
        categorical_cols = param['categorical_cols']
        top_k = param['top_k']
        cuda = torch.cuda.is_available()
        if cuda:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        # miss type 0-7:10%-50%MCAR,20% MAR,MNAR,Region
        for j in [1]:
            all_ARMSE, all_AMAE  = 0, 0
            ARMSE_list, AMAE_list = [],[]
            # run 5 times
            for k in range(1):
                miss_rate, miss_type = util.get_miss_type(j)
                ori_data, miss_data, data_m, label_data, true_data, continuous_cols, nan_data, val_data = data_loader(label_file, miss_rate, categorical_cols, label, miss_type,k)
                values, value_num, value_cat = value_loader(label_file, continuous_cols, label)
                miss_data.columns = values
                miss_data.to_csv(path + 'miss_data_{}.csv'.format(miss_rate), index=None)
                copy_ori_data = ori_data.copy()
                # We learn the encoding rules solely from the raw data and do not use them subsequently.
                ori_code, enc = categorical_to_code(copy_ori_data, value_cat, enc=None)
                # Diffusion_GAN
                ARMSE,AMAE = DiffGANI.get_Diff_acc_RMSE(nan_data, path, miss_rate, miss_data, enc, data_m, categorical_cols, continuous_cols,value_cat, device, param, label_data,values,ori_data,label_num,val_data)
                ARMSE_list.append(ARMSE)
                AMAE_list.append(AMAE)
                all_ARMSE = all_ARMSE + ARMSE
                all_AMAE = all_AMAE + AMAE
            ARMSE, AMAE = all_ARMSE / 1., all_AMAE/1.
            print("dataset：{}， missrate：{}， misstype：{}，ARMSE：{:.4f}，AMAE：{:.4f}".format(
                        data_name, miss_rate, miss_type, ARMSE, AMAE))

        continue

