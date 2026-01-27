import math
import sys
import os
sys.path.append(os.getcwd())
from torch import optim
from model.Discriminator_model import D
from utils.field import CategoricalField, NumericalField
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import random
from model.Learner import train_L_code
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 传入Input_data，将数据-属性上最小值/max-min * 2 - 1
def normalization(data, parameters=None):
    # Parameters
    _, dim = data.shape
    norm_data = data.copy()
    def clamp_data(data):
        return np.where(data > 1, 1, np.where(data < -1, -1, data))
    if parameters is None:
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        # For each dimension
        for i in range(dim):
            # if i == 7 :
            #     print(1)
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i])+ 1e-6)
            norm_data[:, i] = clamp_data(norm_data[:,i])
            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            # if i == 7:
            #     print(1)
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)
            norm_data[:, i] = clamp_data(norm_data[:,i])
        norm_parameters = parameters
    return norm_data, norm_parameters


# feed_data表示为[X1,X2,,.Xn]-->[F1,F21、F22,,,,FN],如果为类别数据使用one-hot向量进行表示
def Data_convert(data, model_name, continuous_cols):
    fields = []
    feed_data = []
    for i, col in enumerate(list(data)):
        # data[i]第i列所有数据
        if i in continuous_cols:
            col2 = NumericalField(model=model_name)
            # 把data传进去
            col2.get_data(data[i])
            fields.append(col2)
            # 获取mean等
            col2.learn()
            # 对第i列数据使用数据标准化方式进行编码,数据-均值/方差
            feed_data.append(col2.convert(np.asarray(data[i])))
        else:
            col1 = CategoricalField("one-hot", noise=None)
            fields.append(col1)
            col1.get_data(data[i])
            col1.learn()
            # 将类别数据使用one-hot向量进行编码
            features = col1.convert(np.asarray(data[i]))
            cols = features.shape[1]
            rows = features.shape[0]
            for j in range(cols):
                feed_data.append(features.T[j])
    feed_data = pd.DataFrame(feed_data).T
    return fields, feed_data



# "Pass in the 'impute_data', the initial dataset, 'M', a list of attribute category names under 'value_cat', and a list for numeric types."
def errorLoss(imputed_data, ori_data, M, value_cat, continuous_cols, enc):
    copy_ori_data = ori_data.copy()
    copy_imputed_data = imputed_data.copy()
    no, dim = copy_imputed_data.shape
    H = np.ones((no, dim))
    # 'H' is set to 0 for all numeric type data.
    for i in continuous_cols:
        H[:, i] = 0
    # In 'data_h', categorical data remains missing, while numeric data is set to 1
    data_h = 1 - (1 - M) * H
    # In 'data_m', numeric data remains missing, while categorical data is set to 1.
    data_m = 1 - (1 - M) * (1 - H)

    if len(value_cat) != 0 and enc is not None:
        copy_imputed_data[value_cat] = enc.transform(copy_imputed_data[value_cat])
        copy_ori_data[value_cat] = enc.transform(copy_ori_data[value_cat])

    imputed_data = copy_imputed_data.values
    ori_data = copy_ori_data.values
    imputed_data = imputed_data.astype(float)
    imputed_data = np.nan_to_num(imputed_data)
    data_m = data_m.astype(float)
    ori_data = ori_data.astype(float)
    ori_data = np.nan_to_num(ori_data)

    # Numeric data is normalized to a range of [-1, 1].
    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)

    # make all data to [-1, 1]

    ARMSE = 0
    AMAE = 0
    cat_ARMSE, cat_AMAE = 0, 0
    num_ARMSE, num_AMAE = 0, 0
    miss_dim = 0
    cat_dim, num_dim = 0, 0
    for i in range(dim):
        ori_get_data = ori_data[:, i]
        imputed_get_data = imputed_data[:, i]
        if i in continuous_cols:
            data_i_m = data_m[:, i]
            if np.sum((1 - data_i_m)) == 0:
                continue
            AR = np.sqrt(np.sum((1 - data_i_m) * ((ori_get_data - imputed_get_data) ** 2)) / np.sum(1 - data_i_m))
            ARMSE = ARMSE + AR
            MAE = np.sum((1 - data_i_m) * np.abs(ori_get_data - imputed_get_data)) / np.sum(1 - data_i_m)
            AMAE = AMAE + MAE
            num_AMAE += MAE
            num_ARMSE += AR
            miss_dim = miss_dim + 1
            num_dim = num_dim + 1
        else:
            data_i_h = data_h[:, i]
            if np.sum((1 - data_i_h)) == 0:
                continue
            equal = (ori_get_data != imputed_get_data).astype('int')

            AR = (np.sum((1 - data_i_h) * equal) / np.sum(1 - data_i_h))
            MAE = np.sum((1 - data_i_h) * equal) / np.sum(1 - data_i_h)
            cat_ARMSE = cat_ARMSE + AR
            cat_AMAE = cat_AMAE + MAE
            ARMSE = ARMSE + AR
            AMAE = AMAE + MAE
            miss_dim = miss_dim + 1
            cat_dim = cat_dim + 1
    cat_ARMSE, cat_AMAE = cat_ARMSE/cat_dim if cat_dim>0 else 0, cat_AMAE/cat_dim if cat_dim>0 else 0
    num_ARMSE, num_AMAE = num_ARMSE/num_dim if num_dim>0 else 0, num_AMAE/num_dim if num_dim>0 else 0
    # print("当前数据集下类别的损失为：ARMSE: {}，AMAE: {}\n数值的损失为：ARMSE: {}，AMAE{}\n".format(cat_ARMSE, cat_AMAE, num_ARMSE, num_AMAE))
    ARMSE = ARMSE / miss_dim
    AMAE = AMAE / miss_dim
    return ARMSE,AMAE

def calculate_error_rate(ori_data, data_m, imputed_data, use_index):
    results = {}
    for column in use_index:
        print(f"Percentage frequencies for column: {column}")
        frequency = ori_data[column].value_counts(normalize=True) * 100
        frequency = frequency.round(2)
        print(frequency)

        ori_column_data = ori_data[column]
        imputed_column_data = imputed_data[column]
        column_index = ori_data.columns.get_loc(column)
        mask = data_m[:, column_index] == 0
        
        ori_values = ori_column_data[mask]
        imputed_values = imputed_column_data[mask]
        
        unique_values = ori_values.unique()
        error_rates = {}
        
        for value in reversed(unique_values):
            correct = (ori_values == value) & (imputed_values == value)
            total = (ori_values == value)
            error_rate = 100 * (1 - correct.sum() / total.sum())
            error_rates[value] = f"{error_rate:.2f}"
        results[column] = error_rates

    for column, rates in results.items():
        print(f"Error rates for column: {column}")
        for value, rate in rates.items():
            print(f"{value}: {rate}")
        print()
    return results

    




def errorLoss_fd(imputed_data, ori_data, M, value_cat, continuous_cols, enc, fd_cols):
    copy_ori_data = ori_data.copy()
    copy_imputed_data = imputed_data.copy()
    no, dim = copy_imputed_data.shape
    H = np.ones((no, dim))
    # H在数值类型数据全设为0
    for i in continuous_cols:
        H[:, i] = 0
    # data_h类别类型上数据保持为缺失状态，数值上数据全为1
    data_h = 1 - (1 - M) * H
    # data_m数值上的保持为缺失状态，类别上数据全为1
    data_m = 1 - (1 - M) * (1 - H)

    if len(value_cat) != 0 and enc is not None:
        copy_imputed_data[value_cat] = enc.transform(copy_imputed_data[value_cat])
        copy_ori_data[value_cat] = enc.transform(copy_ori_data[value_cat])

    imputed_data = copy_imputed_data.values
    ori_data = copy_ori_data.values
    imputed_data = imputed_data.astype(float)
    imputed_data = np.nan_to_num(imputed_data)
    data_m = data_m.astype(float)
    ori_data = ori_data.astype(float)
    ori_data = np.nan_to_num(ori_data)
    # cate_imputed_data表示类别上的数据
    cate_imputed_data = imputed_data * (1 - data_h)
    cate_ori_data = ori_data * (1 - data_h)
    Z = (cate_imputed_data == cate_ori_data)
    Z = Z.astype('int')
    Z = 1 - Z

    # 对数值数据进行归一化为[-1,1]
    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)
    # data_m = M.astype(float)
    cur_num = (1 - data_m) * ori_data - (1 - data_m) * imputed_data
    CORR = cur_num ** 2
    # RMSE = np.sqrt(np.sum(CORR) / np.sum(1 - data_m) + 1e-5)

    nominator = np.sum(CORR) + np.sum(Z)
    # Only for missing values
    denominator = np.sum(1 - M)
    rmse = np.sqrt(nominator / float(denominator))
    mse = nominator / float(denominator)
    # print("类别损失为：{}  数值损失为：{}".format(np.sum(Z), np.sum(CORR)))
    # 遍历每一列求其RMSE或者MAE
    ARMSE = 0
    AMAE = 0
    miss_dim = 0
    for i in range(dim):
        ori_get_data = ori_data[:, i]
        imputed_get_data = imputed_data[:, i]
        if i in fd_cols:
            data_i_h = data_h[:, i]
            if np.sum((1 - data_i_h)) == 0:
                continue
            equal = (ori_get_data != imputed_get_data).astype('int')
            AR = (np.sum((1 - data_i_h) * equal) / np.sum(1 - data_i_h))
            MAR = np.sum((1 - data_i_h) * equal) / np.sum(1 - data_i_h)
            ARMSE = ARMSE + AR
            AMAE = AMAE + MAR
            miss_dim = miss_dim + 1
        # if AR > 1:
        #     print(1)
    ARMSE = ARMSE / miss_dim
    AMAE = AMAE / miss_dim
    return ARMSE, AMAE
# 传入解码后的具体数值current_data，以及类别的属性名，将current_data中类别的属性值变为int类型的分类数据
def labelCode(data, value_cat, enc):
    data[value_cat] = enc.transform(data[value_cat])
    return data


# 获取Imputed Data
def concatValue(current_data, miss_data, m):
    current_data = current_data.values
    miss_data = miss_data.values
    new_data = miss_data * m + current_data * (1 - m)
    return pd.DataFrame(new_data)

def resver_value(data, value_cat, enc):
    data[value_cat] = enc.inverse_transform(data[value_cat])
    return data


# 传入解码后数据，filed每个属性编码方式,value_cat类别属性名列表,ori_data_x GroundTruth，data_m缺失数据的位置M
# 输出Inputted Data归一化后的数据
def reconvert_data(x_, fields, value_cat, values, miss_data_x, data_m, enc):
    current_data = []
    current_ind = 0
    for i in range(len(fields)):
        dim = fields[i].dim()
        # 将decoder之后的数据再进行编码回原来值
        data_transept = x_[:, current_ind:(current_ind + dim)].cpu().detach().numpy()
        # 将数据根据最开始的编码还原,即x*self.sigma + self.mu
        current_data.append(pd.DataFrame(fields[i].reverse(data_transept)))
        current_ind = current_ind + dim
    current_data = pd.concat(current_data, axis=1)
    current_data.columns = values

    if value_cat:
        # for column in miss_data_x.columns:
        #     random_value = "Null"
        #     while random_value == "Null":
        #         random_value = random.choice(miss_data_x[column])
        #     miss_data_x[column] = miss_data_x[column].replace("Null", random_value)
        # current_data = labelCode(current_data, value_cat, enc)  # current_data解码后的具体数值，分类的转为int类型
        miss_data_x = labelCode(miss_data_x, value_cat, enc)
        current_data = concatValue(current_data, miss_data_x, data_m)  # 获取Imputed Data
        current_data.columns = values
        current_data = resver_value(current_data, value_cat, enc)
    else:
        current_data = concatValue(current_data, miss_data_x, data_m)
    return current_data


# 对数值类数据归一化数据
def Num_Normalize(res_data, categorical_cols):
    for index, name in enumerate(res_data.columns):
        if index not in categorical_cols:
            min_val = res_data[name].min()
            max_val = res_data[name].max()

            def normalize(x):
                if max_val == min_val:
                    return x
                else:
                    return (x - min_val) / (max_val - min_val) * 2 - 1

            res_data[name] = res_data[name].apply(normalize)
    return res_data


# 对分类数据归一化数据
def Cat_Normalize(res_data, categorical_cols):
    for index, name in enumerate(res_data.columns):
        if index in categorical_cols:
            min_val = res_data[name].min()
            max_val = res_data[name].max()

            def normalize(x):
                return (x - min_val) / (max_val - min_val) * 2 - 1

            res_data[name] = res_data[name].apply(normalize)
    return res_data


# 计算元组间的相似性
def get_tuple_sim(tuple_value, sim_tuple_value, categorical_cols, data_m, ori_index, sim_index):
    num_sim = 0
    cat_sim = 0
    nan_sim = 0
    for index in range(len(tuple_value)):
        if data_m[ori_index, index] == 0 or data_m[sim_index, index] == 0:
            nan_sim = nan_sim + 1
            continue
        if index not in categorical_cols:
            num_sim = num_sim + math.sqrt((tuple_value[index] - sim_tuple_value[index]) ** 2)
        else:
            if tuple_value[index] != sim_tuple_value[index]:
                cat_sim = cat_sim + 1
    return num_sim + cat_sim + nan_sim


# 初始的注意力机制，即使用最相似的单元格填充数据x=(1 − mi) ⊙ sim(xi) + mi ⊙ xi
def init_attn(miss_data, data_m, categorical_cols):
    res_data = miss_data.copy()
    # 获取每列属性的数据
    attr_list_map = {}
    for col_name in res_data.columns:
        attr_list_map[col_name] = res_data[col_name].value_counts()
    data_num = len(res_data)
    # 对res_data归一化为[-1,1]
    res_data = Num_Normalize(res_data, categorical_cols)
    data_list = []
    # 存放每个元组与其他元组的相似度
    sim_data = {i: {} for i in range(data_num)}
    # 为每个元组找到sim(xi)
    for index, row in res_data.iterrows():
        cur_data_list = row.values
        data_list.append(cur_data_list.tolist())
    # 随机选中进行匹配的元组
    for index, tuple_value in enumerate(data_list):
        random_list = random.sample([x for x in range(0, data_num) if x != index], 30)
        for sim_index in random_list:
            sim_tuple_value = data_list[sim_index]
            sim_value = get_tuple_sim(tuple_value, sim_tuple_value, categorical_cols, data_m, index, sim_index)
            sim_data[index][sim_index] = sim_value
    sim_list = {}  # 存放每个元组使用sim填充后的数据
    for key in sim_data:
        cur_sim = sim_data[key]
        # 设置一个反转的dict
        revert_sim = {}
        for k in cur_sim:
            revert_sim[cur_sim[k]] = k
        new_cur_sim = [x for x in cur_sim.values() if x > 0]
        sorted_list = sorted(new_cur_sim)
        cur_list = []
        # 取前十个最相似的元组填充
        for i in sorted_list:
            smallest_index = revert_sim[i]
            sim_tuple = miss_data.iloc[smallest_index]
            cur_list.append(sim_tuple.tolist())
        sim_list[key] = cur_list
    a = 1 - data_m
    X = miss_data.copy().values
    for index, list_col in enumerate(a):
        # if index == 163:
        #     print(1)
        for cur_index, value in enumerate(list_col):
            if value == 1:
                cur_tup_list = sim_list[index]
                for tup in cur_tup_list:
                    if tup[cur_index] != "Null":
                        X[index][cur_index] = tup[cur_index]
                        break
                if X[index][cur_index] == "Null":
                    k = miss_data.columns[cur_index]
                    for val in attr_list_map[k].index.tolist():
                        if val != "Null":
                            X[index][cur_index] = val
                            break
    X = pd.DataFrame(X)
    X.columns = miss_data.columns
    return X


# 传如data_m数组，每行的编码方式，输出编码后的tensor的M
def get_M_by_data_m(data_m, filed, device):
    data_m = pd.DataFrame(data_m)
    # 扩充data_m
    M = data_m.copy()
    begin = 0
    for index, i in enumerate(filed):
        if i.data_type != "Numerical Data":
            one_hot_len = len(i.rev_dict)
            new_col = pd.concat([data_m.iloc[:, index]] * one_hot_len, axis=1)
            M = M.iloc[:, :begin].join(new_col).join(M.iloc[:, begin + 1:])
            begin = begin + one_hot_len
        else:
            begin = begin + 1
    M = torch.tensor(M.values, dtype=torch.float).to(device)
    return M

# Use attention to retrieve the padded data.
def init_attn_2(corr_map, miss_data, data_m, categorical_cols, enc, value_cat, device, top_k):
    corr_map_copy = None
    corr_cur = None
    corr_list = None
    miss_data_code = miss_data.copy()
    values = miss_data.columns
    attr_list_map = {}
    con_cols = []
    co_all = miss_data.columns
    for index, val in enumerate(co_all):
        if index not in categorical_cols:
            con_cols.append(index)
    for col_name in miss_data.columns:
        attr_list_map[col_name] = miss_data[col_name].value_counts()

    for i in categorical_cols:
        for val in attr_list_map[miss_data.columns[i]].index.tolist():
            if val != "Null":
                miss_data_code[miss_data.columns[i]] = miss_data_code[miss_data.columns[i]].apply(
                    lambda x: val if x == 'Null' else x)
    miss_data_code, enc = categorical_to_code(miss_data_code, value_cat, enc)
    miss_data_code.columns = [x for x in range(miss_data_code.shape[1])]
    filed, miss_data_code = Data_convert(miss_data_code, "mean_std", con_cols)
    # If the computer is well-configured, everything can be loaded onto the GPU.
    if miss_data_code.shape[0] < 1000:
        miss_data_code = torch.tensor(miss_data_code.values, dtype=torch.float).to(device)
    else:
        miss_data_code = torch.tensor(miss_data_code.values, dtype=torch.float).cpu()
    data_m = pd.DataFrame(data_m)
    M = data_m.copy()
    begin = 0
    begin_list = [0]
    if corr_map is not None:
        corr_map = pd.DataFrame(corr_map)
        corr_map_copy = corr_map.copy()
    for index, i in enumerate(filed):
        if i.data_type != "Numerical Data":
            one_hot_len = len(i.rev_dict)
            new_col = pd.concat([data_m.iloc[:, index]] * one_hot_len, axis=1)
            M = M.iloc[:, :begin].join(new_col).join(M.iloc[:, begin + 1:])
            if corr_map is not None:
                new_col_corr = pd.concat([corr_map.iloc[:, index]] * one_hot_len, axis=1)
                corr_map_copy = corr_map_copy.iloc[:, :begin].join(new_col_corr).join(corr_map_copy.iloc[:, begin + 1:])
            begin = begin + one_hot_len
        else:
            begin = begin + 1
        begin_list.append(begin)
    Corr_map = None
    if corr_map is not None:
        Corr_map = torch.tensor(corr_map_copy.values, dtype=torch.float).to(device)
    M = torch.tensor(M.values, dtype=torch.float).to(device)
    if M.size()[0] >= 1000:
        Corr_map = Corr_map.cpu() if Corr_map is not None else None
        M = M.cpu()
    ori_miss_data_code = miss_data_code * M
    miss_data_code = ori_miss_data_code.clone()
    true_data_index_col = {}
    miss_data_index_col = {}
    data_M = data_m.values
    for i in range(data_M.shape[1]):
        indices = np.where(data_M[:, i] == 1)[0]
        miss_in = np.where(data_M[:, i] == 0)[0]
        true_data_index_col[i] = indices.tolist()
        miss_data_index_col[i] = miss_in.tolist()
    if corr_map is not None:
        corr_list = []
        for col in range(len(co_all)):
            cur_col_corr = Corr_map[col]
            cur_col_corr = cur_col_corr.repeat(miss_data_code.shape[0], 1)
            corr_cur = torch.matmul(miss_data_code * cur_col_corr, miss_data_code.T)
            corr_cur.diagonal(offset=0).fill_(float('-inf'))
            corr_cur[miss_data_index_col[col], :] = float('-inf')
            corr_cur = torch.softmax(corr_cur, dim=0)
            corr_cur = torch.where(torch.isnan(corr_cur), torch.zeros_like(corr_cur), corr_cur)
            k = int(corr_cur.shape[0] * top_k)
            top_k_tensor = torch.topk(corr_cur, k=k, dim=0).values[-1, :].unsqueeze(0)
            top_k_tensor = top_k_tensor.expand_as(corr_cur)
            corr_cur[corr_cur < top_k_tensor] = 0
            row_sum = torch.sum(corr_cur, dim=0).unsqueeze(0)
            row_sum = torch.where(row_sum == 0, torch.tensor(1e-7).to(row_sum.device), row_sum)
            corr_cur = corr_cur / row_sum
            corr_list.append(corr_cur)
    else:
        corr_cur = torch.matmul(miss_data_code, miss_data_code.T)
        corr_cur.diagonal(offset=0).fill_(float('-inf'))
        corr_cur = torch.softmax(corr_cur, dim=0)
        k = int(corr_cur.shape[0] * top_k)
        top_tensor = torch.topk(corr_cur, k=k, dim=0).values[:, -1].unsqueeze(1)
        top_tensor = top_tensor.expand_as(corr_cur)
        corr_cur[corr_cur < top_tensor] = 0
        row_sum = torch.sum(corr_cur, dim=0)
        corr_cur = corr_cur / row_sum.unsqueeze(0)
    p = 1 - data_M
    impute_data = miss_data.values
    impute_data_code = ori_miss_data_code.clone()
    impute_cell_acc = np.ones_like(data_M).astype(float)
    if corr_map is None:
        impute_code = torch.matmul(corr_cur.T, impute_data_code)
    else:
        corr_list_code = []
        for index, corr_cur in enumerate(corr_list):
            if filed[index].data_type == "Numerical Data":
                cur_M = M[:, begin_list[index]]
                cur_code = impute_data_code[:, begin_list[index]]
                cur_code = torch.matmul(corr_cur.T, cur_code).unsqueeze(-1)
            else:
                cur_M = M[:, begin_list[index]:begin_list[index+1]]
                cur_code = impute_data_code[:, begin_list[index]:begin_list[index+1]]
                cur_code = torch.matmul(corr_cur.T, cur_code)
            corr_list_code.append(cur_code)
        impute_code = torch.cat(corr_list_code, dim=1)
    new_code = impute_code * (1 - M) + impute_data_code * M
    impute_data = reconvert_data(new_code, filed, value_cat, values, miss_data, data_m, enc)
    impute_data = pd.DataFrame(impute_data)
    impute_data.columns = values
    new_code = new_code.to(device)
    return impute_data, new_code


def get_miss_type(j):
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



# (n*1) (n*1)
def get_cell_acc(encode_code, corr_cur):
    avg = torch.matmul(encode_code.T, corr_cur).cpu().numpy()
    encode_code_numpy = encode_code.cpu().numpy()
    corr_cur_numpy = encode_code.cpu().numpy()
    # 计算差值
    diff = [x - avg for x in encode_code_numpy]
    # 计算平方差值
    squared_diff = np.array([x ** 2 for x in diff])
    # 计算方差
    variance = np.sum(squared_diff * corr_cur_numpy)
    cell_acc = np.exp(-variance)
    return cell_acc
# 将类别数据进行编码
def categorical_to_code(miss_data_x, value_cat, enc=None, encoding_mode='ordinal', emb_dim=8):
    if len(value_cat) == 0:
        return miss_data_x, None

    if encoding_mode == 'ordinal':
        # 原始的 ordinal 编码方式
        if enc is None:
            # 将类别进行编码为数字
            # enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            enc = OrdinalEncoder()
            enc.fit(miss_data_x[value_cat])
        attr_list_map = {}
        for col_name in miss_data_x.columns:
            attr_list_map[col_name] = miss_data_x[col_name].value_counts()

        miss_data_x[value_cat] = enc.transform(miss_data_x[value_cat])
        sim_data_x = pd.DataFrame(miss_data_x)
        return sim_data_x, enc

    elif encoding_mode == 'embedding':
        # Embedding lookup 编码方式
        import torch
        import torch.nn as nn

        # 为每个分类列创建 embedding 信息
        embedding_info = {}
        encoded_data = miss_data_x.copy()

        for col in value_cat:
            # 将列转换为 category 类型
            cat_series = miss_data_x[col].astype("category")
            cardinality = cat_series.nunique()

            # 获取编码后的数值
            codes = cat_series.cat.codes.values

            # 创建 embedding 层
            embedding_layer = nn.Embedding(cardinality, emb_dim)

            # 将编码后的数据转换为 embedding 向量
            codes_tensor = torch.tensor(codes, dtype=torch.long)
            embeddings = embedding_layer(codes_tensor).detach().numpy()

            # 将 embedding 向量存储为新的列
            for i in range(emb_dim):
                encoded_data[f"{col}_emb_{i}"] = embeddings[:, i]

            # 保存 embedding 信息
            embedding_info[col] = {
                'cardinality': cardinality,
                'embedding_dim': emb_dim,
                'categories': cat_series.cat.categories.tolist(),
                'codes': codes
            }

            # 移除原始分类列
            encoded_data = encoded_data.drop(col, axis=1)

        # 创建一个兼容的编码器对象，用于向后兼容性
        class EmbeddingEncoder:
            def __init__(self, embedding_info):
                self.embedding_info = embedding_info
                self.categories_ = {}
                for col, info in embedding_info.items():
                    self.categories_[col] = info['categories']

            def transform(self, data):
                # 对于兼容性，transform 方法的行为应该与 OrdinalEncoder 类似
                # 即接受分类列数据，返回编码后的数值数组
                if isinstance(data, pd.DataFrame):
                    # 如果传入的是 DataFrame（通常是 miss_data_x[value_cat]），返回数值编码数组
                    # 保持与 OrdinalEncoder 相同的接口：返回与输入形状相同的数组
                    result = np.zeros(data.shape, dtype=int)
                    for i, col in enumerate(data.columns):
                        if col in self.embedding_info:
                            cat_series = data[col].astype("category")
                            cat_series = cat_series.cat.set_categories(self.categories_[col])
                            codes = cat_series.cat.codes.values
                            codes = np.clip(codes, 0, self.embedding_info[col]['cardinality'] - 1)  # 处理未知类别
                            result[:, i] = codes
                    return result
                else:
                    # 如果传入的是 Series 或数组，返回编码后的数值
                    # 这里简化处理，假设是单个列的数据
                    col_name = list(self.embedding_info.keys())[0]  # 假设只有一个分类列
                    if hasattr(data, 'values'):
                        data_array = data.values
                    else:
                        data_array = np.array(data)
                    cat_series = pd.Series(data_array.flatten()).astype("category")
                    cat_series = cat_series.cat.set_categories(self.categories_[col_name])
                    codes = cat_series.cat.codes.values
                    codes = np.clip(codes, 0, self.embedding_info[col_name]['cardinality'] - 1)
                    return codes.reshape(-1, 1)

            def inverse_transform(self, data):
                # 对于 embedding 编码，我们无法真正逆转换，因为 embedding 是连续向量
                # 这里返回最接近的类别标签（简化实现）
                if isinstance(data, pd.DataFrame):
                    decoded_data = data.copy()
                    for col in self.embedding_info.keys():
                        if col in data.columns:
                            # 对于 embedding 编码，我们使用原始编码值来近似逆转换
                            # 这里简化处理，直接使用编码值作为类别索引
                            codes = data[col].values.astype(int)
                            codes = np.clip(codes, 0, len(self.categories_[col]) - 1)
                            decoded_data[col] = [self.categories_[col][code] for code in codes]
                    return decoded_data
                else:
                    # 简化处理，假设是单个列的数据
                    col_name = list(self.embedding_info.keys())[0]
                    codes = np.array(data).astype(int).flatten()
                    codes = np.clip(codes, 0, len(self.categories_[col_name]) - 1)
                    return np.array([self.categories_[col_name][code] for code in codes]).reshape(-1, 1)

        compatible_enc = EmbeddingEncoder(embedding_info)
        return encoded_data, compatible_enc

    elif encoding_mode == 'onehot':
        # One-hot 编码方式
        from sklearn.preprocessing import OneHotEncoder

        if enc is None:
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            enc.fit(miss_data_x[value_cat])

        # 进行 one-hot 编码
        onehot_encoded = enc.transform(miss_data_x[value_cat])

        # 创建新的 DataFrame
        encoded_data = miss_data_x.drop(value_cat, axis=1)

        # 添加 one-hot 编码的列
        feature_names = enc.get_feature_names_out(value_cat)
        onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names, index=miss_data_x.index)
        encoded_data = pd.concat([encoded_data, onehot_df], axis=1)

        # 创建一个兼容的编码器对象，用于向后兼容性
        class OneHotEncoderWrapper:
            def __init__(self, onehot_encoder, original_columns):
                self.encoder = onehot_encoder
                self.original_columns = original_columns
                self.feature_names = onehot_encoder.get_feature_names_out(original_columns)

            def transform(self, data):
                # 对于兼容性，返回一个与原始列数相同的占位符数组
                # 因为 one-hot 编码改变了数据结构，后续调用应该不需要再次转换
                if isinstance(data, pd.DataFrame):
                    return np.zeros((data.shape[0], len(self.original_columns)))
                else:
                    # 假设是数组输入
                    data_length = data.shape[0] if hasattr(data, 'shape') else len(data)
                    return np.zeros((data_length, len(self.original_columns)))

            def inverse_transform(self, data):
                # One-hot 编码不支持精确逆转换，这里返回占位符
                return np.zeros((data.shape[0], len(self.original_columns)))

        compatible_enc = OneHotEncoderWrapper(enc, value_cat)
        return encoded_data, compatible_enc

    else:
        raise ValueError(f"Unsupported encoding_mode: {encoding_mode}. Choose from 'ordinal', 'embedding', 'onehot'")


# 获取数据的均值方差
def get_number_data_mu_var(zero_feed_data_code, M_tensor, fields, device):
    if M_tensor is not None:
        M_numpy = M_tensor.cpu().numpy()
    zero_feed_data = pd.DataFrame(np.array(zero_feed_data_code.cpu()))
    begin = 0
    mu_var_list = []
    for col_num, filed in enumerate(fields):
        cur_dict = {}
        if filed.data_type != "Categorical Data":
            if M_tensor is not None:
                no_miss_index = np.array(np.where(M_numpy[:, begin] == 1)).flatten()
                no_miss_data = zero_feed_data.iloc[no_miss_index, begin]
            else:
                no_miss_data = zero_feed_data.iloc[:, begin]
            mu = np.mean(no_miss_data)
            var = np.var(no_miss_data)
            cur_dict['mu'] = torch.tensor(mu).to(device)
            cur_dict['var'] = torch.tensor(var).to(device)
            begin = begin + 1
        else:
            begin = begin + filed.dim()
            cur_dict['mu'] = 0
            cur_dict['var'] = 0
        mu_var_list.append(cur_dict)
    return mu_var_list



def sample_x(x, batch_size):
    rows_to_change = list(np.random.choice(x.size()[0], batch_size, replace=False))
    sample_data = x[rows_to_change, :]
    return sample_data, rows_to_change

# 获取填充后数据均值方差
def get_impute_data_mu_var(decoder_z_impute, fields):
    begin = 0
    mu_var_list = []
    for col_num, filed in enumerate(fields):
        cur_dict = {}
        if filed.data_type != "Categorical Data":
            no_miss_data = decoder_z_impute[:, begin]
            mu = torch.mean(no_miss_data)
            var = torch.var(no_miss_data)
            cur_dict['mu'] = mu
            cur_dict['var'] = var
            begin = begin + 1
        else:
            begin = begin + filed.dim()
            cur_dict['mu'] = 'null'
            cur_dict['var'] = 'null'
        mu_var_list.append(cur_dict)
    return mu_var_list

def get_num_mu_var_loss(new_data_num_var, num_mu_var):
    all_loss = 0
    col = 0
    for col_num, cur_dict in enumerate(num_mu_var):
        if new_data_num_var[col_num]['mu'] == 'null':
            continue
        new_data_mu = new_data_num_var[col_num]['mu']
        new_data_var = new_data_num_var[col_num]['var']
        ori_data_mu = cur_dict['mu']
        ori_data_var = cur_dict['var']
        col_kl_val = torch.log(new_data_var/ori_data_var) + 0.5 * ((torch.pow(ori_data_var, 2) +
                                                                   torch.pow(ori_data_mu-new_data_mu, 2)) / torch.pow(new_data_var, 2) - 1)
        all_loss = all_loss + col_kl_val
        col = col + 1
    return all_loss/col

def get_valid_data_index(data_m, discriminator, impute_data_code, device):
    valid_data_index = [i for i in range(len(data_m)) if all(val == 1 for val in data_m[i])]
    if len(valid_data_index) < int(0.2 * data_m.shape[0]):
        curr_dis = D(discriminator.input_dim, discriminator.latent_dim, discriminator.out_dim).to(device)
        optimizer_D_cur = optim.Adam(curr_dis.parameters(), lr=0.002)
        curr_dis.train()
        m_data = torch.tensor(data_m).float().to(device)
        for i in range(1000):
            optimizer_D_cur.zero_grad()
            D_pro = curr_dis(impute_data_code)
            loss = -torch.mean(m_data * torch.log(D_pro + 1e-8) + (1 - m_data) * torch.log(1 - D_pro + 1e-8))
            loss.backward()
            optimizer_D_cur.step()
        D_pro = curr_dis(impute_data_code)
        # miss_data_numpy = curr_dis(zero_feed_data).cpu().detach().numpy()
        # D_pro_numpy = D_pro.cpu().detach().numpy()
        # m_data_numpy = m_data.cpu().detach().numpy()
        sum_tensor = D_pro.sum(dim=1)
        sorted_tensor, indices = torch.sort(sum_tensor)
        top_indices = indices[:int(0.2 * data_m.shape[0])].cpu().numpy()
        valid_data_index = np.union1d(valid_data_index, top_indices)
    else:
        valid_data_index = random.sample(valid_data_index, int(0.2 * data_m.shape[0]))
    return valid_data_index

def test_impute_data_rmse(x_code, fields, value_cat, values, miss_data_x, data_m, enc, ori_data, continuous_cols):
    # 计算直接用attention填充的RMSE和ACC
    impute_data = reconvert_data(x_code, fields, value_cat, values, miss_data_x, data_m, enc)
    impute_data = pd.DataFrame(impute_data)
    impute_data.columns = values
    # 通过enc将类别数据还原
    # if value_cat:
    #     impute_data[value_cat] = enc.inverse_transform(impute_data[value_cat])
        # 传入input——data,初始的数据，M，类别的属性，数值类型属性，返回RMSE损失
    rmse, mse = errorLoss(impute_data, ori_data, data_m, value_cat, continuous_cols, enc)
    return rmse, mse
def test_fd_data_rmse(x_code, fields, value_cat, values, miss_data_x, data_m, enc, ori_data, continuous_cols, fd_cols):
    # 计算直接用attention填充的RMSE和ACC
    impute_data = reconvert_data(x_code, fields, value_cat, values, miss_data_x, data_m, enc)
    impute_data = pd.DataFrame(impute_data)
    impute_data.columns = values
    # 通过enc将类别数据还原
    # if value_cat:
    #     impute_data[value_cat] = enc.inverse_transform(impute_data[value_cat])
    # 传入input——data,初始的数据，M，类别的属性，数值类型属性，返回RMSE损失
    rmse = errorLoss_fd(impute_data, ori_data, data_m, value_cat, continuous_cols, enc, fd_cols)
    return rmse
def test_impute_data_acc(x, valid_data_index, label_data_code, train_data_index,label_num, device):
    x_valid = x[valid_data_index]
    y_valid = label_data_code[valid_data_index]
    x_train = x[train_data_index]
    y_train = label_data_code[train_data_index]
    L_loss, Acc = train_L_code(1000, x_train, y_train, x_valid, y_valid, label_num, device)
    return Acc

def test_impute_data_Acc(x, label_data_code, val_data, label_num, value_cat, continuous_cols, enc, device):
    x_train = x.detach()
    y_train = label_data_code
    val_x = val_data.iloc[:, :-1]
    cat_to_code_data, enc = categorical_to_code(val_x.copy(), value_cat, enc)
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    fields_1, x_val = Data_convert(cat_to_code_data, "mean_std", continuous_cols)
    x_val = torch.tensor(x_val.values, dtype=torch.float).to(device)
    val_y = val_data.iloc[:, -1]
    y_val = torch.FloatTensor(val_y.values).to(device)
    _, acc = train_L_code(1000, x_train, y_train, x_val, y_val, label_num, device)
    return acc


def sort_corr(corr_map):
    sort_dict = {}
    for index,corr_data in enumerate(corr_map):
        sorted_indices = sorted(range(len(corr_data)), key=lambda x: corr_data[x], reverse=True)
        sort_dict[index] = sorted_indices
    return sort_dict

# Randomly order tuples based on a selected numerical attribute, then choose a sequence of continuous tuples.
# Randomly inject missing values into other attributes of these tuples.
def MAR(data, continuous_cols, miss_seed):
    np.random.seed(miss_seed)
    new_data = data.copy()
    if len(continuous_cols) == 0:
        continuous_cols = [i for i in range(data.shape[1])]
    choose_num_index = np.random.choice(continuous_cols)
    sorted_indices = np.argsort(new_data.iloc[:, choose_num_index])
    sorted_data = new_data.iloc[sorted_indices]

    # Calculate the number of tuples that need missing values injected
    mar_missing_count = int(len(sorted_data) * 0.4)
    start_index = np.random.randint(0, len(sorted_data) - mar_missing_count + 1)
    selected_data = sorted_data[start_index:start_index + mar_missing_count]
    mask = np.random.choice([True, False], size=selected_data.shape, p=[0.5, 0.5])
    mask[:,choose_num_index] = False
    mask = pd.DataFrame(mask,index=selected_data.index, columns=selected_data.columns)
    selected_data[mask] = np.nan

    sorted_data[start_index:start_index + mar_missing_count] = selected_data
    df = sorted_data.sort_index()
    data_m = np.zeros(df.shape)
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Iterate through values at each position
        for i, value in enumerate(row):
            # Check if the value is NaN
            if pd.isna(value):
                data_m[index, i] = 0
            else:
                data_m[index, i] = 1
    return data_m


def MNAR(data, continuous_cols, categorical_cols, miss_seed):
    np.random.seed(miss_seed)
    mnar_data = data.copy()
    data_m = np.ones(mnar_data.values.shape)
    for i in range(mnar_data.shape[1]):
        if i in categorical_cols:
            mask = np.random.choice([True, False], size=data_m.shape[0], p=[0.2, 0.8])
            data_m[mask, i] = 0
        else:
            # 计算中位数
            median = np.median(mnar_data.iloc[:, i])
            # 生成布尔掩码
            indices_below_median = np.where(mnar_data.iloc[:, i] <= median)
            num_indices = len(indices_below_median[0])
            num_to_change = int(num_indices * 0.4)
            random_indices = np.random.choice(indices_below_median[0], num_to_change, replace=False)
            # 将这些索引的值置为0
            data_m[random_indices, i] = 0
            # 将小于中位数的数据以0.4的概率变为0
            # data_m[mask & (mnar_data[:, i] < median)] = 0
    # 找出所有0的位置
    # zero_positions = np.where(data_m == 0)

    # 选择50%的位置
    # num_zeros = len(zero_positions[0])
    # num_to_change = int(num_zeros * 0.2)
    # random_indices = np.random.choice(num_zeros, num_to_change, replace=False)

    # 将这些位置的值改为1
    # data_m[zero_positions[0][random_indices], zero_positions[1][random_indices]] = 1
    return data_m



def Region(data, miss_seed):
    np.random.seed(miss_seed)
    data_numpy = data.values
    total_elements = data_numpy.size
    missing_elements = int(total_elements * 0.2)

    # 计算可能的最大区域
    max_rows = missing_elements
    max_cols = 1

    while max_rows > data_numpy.shape[0]:
        max_rows //= 2
        max_cols = missing_elements // max_rows

    # 随机选择起始位置
    start_row = np.random.randint(0, data_numpy.shape[0] - max_rows + 1)
    start_col = np.random.randint(0, data_numpy.shape[1] - max_cols + 1)
    data_m = np.ones((data_numpy.shape[0],data_numpy.shape[1]))
    # 将选定区域设置为np.nan
    data_m[start_row:start_row + max_rows, start_col:start_col + max_cols] = 0
    return data_m


def get_down_acc(impute_data_code, label_data, test_data, value_cat, continuous_cols, enc, seed):
    test_x = test_data.iloc[:, :-1]
    impute_data_code.columns = test_x.columns
    train_data = pd.concat([impute_data_code, test_x], axis=0)
    cat_to_code_data, enc = categorical_to_code(train_data.copy(), value_cat, enc)
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    fields_1,  x = Data_convert(cat_to_code_data, "mean_std", continuous_cols)
    x = x.values

    x_train = x[:impute_data_code.shape[0], :]
    y_train = label_data.values.ravel()

    x_test = x[impute_data_code.shape[0]:, :]
    y_test = test_data.iloc[:, -1].values

    # Training a RandomForest Classifier
    classifier = RandomForestClassifier(random_state=seed)
    classifier.fit(x_train, y_train)

    # Predicting the test set results
    y_pred = classifier.predict(x_test)

    # Calculating the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


