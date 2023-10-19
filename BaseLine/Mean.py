import numpy as np
import pandas as pd


def fill_data_mean(miss_data, con_cols, cat_cols):
    for i in range(miss_data.shape[1]):
        if i in con_cols:
            miss_data[:, i] = mean_fill(miss_data[:, i])
        else:
            miss_data[:, i] = zhong_fill(miss_data[:, i])
    return miss_data


def mean_fill(data):
    all = 0
    for i in data:
        if not np.isnan(i):
            all = all + i
    mean = all / data.shape[0]
    for index,i in enumerate(data):
        if np.isnan(i):
            data[index] = mean
    return data

def zhong_fill(data):
    miss_data = pd.DataFrame(data)
    attr_list_map = miss_data.value_counts()
    zhong_shu = attr_list_map.index.tolist()[0][0]
    miss_data.fillna(zhong_shu, inplace=True)
    data = miss_data.values
    data = np.squeeze(data)
    return data


