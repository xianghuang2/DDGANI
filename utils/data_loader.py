# coding=utf-8
import csv

import numpy as np
import pandas as pd
from pandas import DataFrame
import random

from sklearn.model_selection import train_test_split

import util



# 生成随机的矩阵
def binary_sampler(p, rows, cols, k):
    # if k == 0:
    #     np.random.seed(42)
    # else:
    #     np.random.seed(2)
    uni_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (uni_random_matrix < p)
    data1 = DataFrame(binary_random_matrix, index=None)
    # data1.to_csv('data/what_0.csv', index=False)  # 哪些是替换的
    return binary_random_matrix


# Load missing data
def data_loader(file_name, miss_rate, categorical_cols, label, miss_type, k):
    # Load data
    data_x = pd.read_csv(file_name)
    columns = data_x.columns
    n = len(data_x)
    data_x = data_x.sample(frac=1).reset_index(drop=True)
    label_column = data_x.iloc[:, -1]
    con_cols = []
    if label != "False":
        data_x = data_x.iloc[:, :-1]
    no, dim = data_x.shape
    cols = [i for i in range(dim)]
    for i in cols:
        if i not in categorical_cols:
            con_cols.append(i)
    if miss_type == "MCAR":
        data_m = binary_sampler(1 - miss_rate, no, dim, k)
    elif miss_type == "MAR":
        data_m = util.MAR(data_x, con_cols)
    elif miss_type == "MNAR":
        data_m = util.MNAR(data_x, con_cols, categorical_cols)
    elif miss_type == "Region":
        data_m = util.Region(data_x)
    true_data = [i for i in range(len(data_m)) if all(val == 1 for val in data_m[i])]
    if len(true_data) == 0:
        true_data = None
    else:
        true_data = data_x.loc[true_data]
    sim_data_x = data_x.copy()
    sim_data_x[data_m == 0] = np.nan
    nan_data = sim_data_x.values
    miss_data_x = sim_data_x.copy()
    for i in cols:
        if i not in categorical_cols:
            miss_data_x.iloc[:, i].fillna(0, inplace=True)
        else:
            miss_data_x.iloc[:, i].fillna("Null", inplace=True)
    return data_x, miss_data_x, data_m, pd.DataFrame(label_column), true_data, con_cols, nan_data


# 获取数据的类别属性列以及数值列
def get_categorical_columns(file_name, categorical_num):
    df = pd.read_csv(file_name)
    df.fillna(0, inplace=True)
    categorical_columns = []
    numerical_columns = []
    data_num = len(df)
    dict_data = df.to_dict('dict')  # 字典，键是列名，值是数据列表
    for index, value in enumerate(dict_data):
        a = df.columns[index]
        if df.columns[index] == "label":
            break
        cur_dict = dict_data[value]
        cur_set = set()
        for cur_value in cur_dict.values():
            cur_set.add(cur_value)
        if len(cur_set) >= categorical_num and isinstance(next(iter(cur_set)), (int, float)):
            numerical_columns.append(index)
        else:
            categorical_columns.append(index)
    return categorical_columns, numerical_columns


# 获取类别数据属性名，数值数据的属性名
def value_loader(file_name, continuous_cols, label):
    data_x = pd.read_csv(file_name)
    if label != "False":
        data_x = data_x.iloc[:, :-1]
    col = data_x.shape[1]
    fields = data_x.columns.values
    values = fields.tolist()
    value_num = []
    value_cat = []
    for i in range(col):
        if i in continuous_cols:
            value_num.append(values[i])
        else:
            value_cat.append(values[i])
    return values, value_num, value_cat


# 将最后一列数据设置为学习器的标签
def set_label(file_name, label_file_path, label):
    new_file_path = label_file_path + "label_data.csv"
    df = pd.read_csv(file_name)
    if label == "False":
        label_num = 0
        df.to_csv(new_file_path, index=False)
        return new_file_path, label_num
    else:
        label_column = df.iloc[:, -1]
        df = df.iloc[:, :-1]
        df['label'] = label_column
        unique_values = df['label'].unique()
        label_type = "cat"
        label_map = {value: i for i, value in enumerate(unique_values)}
        df['label'] = df['label'].replace(label_map)
        label_num = len(unique_values)
        df.to_csv(new_file_path, index=False)
        return new_file_path, label_num

