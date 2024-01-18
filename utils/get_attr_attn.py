
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / (min((kcorr-1), (rcorr-1))+1e5))


def get_tongji_attn_map(impute_data, method, isNull):
    if isNull:
        pearson_corr = impute_data
    else:
        pearson_corr = impute_data.corr(method=method)
    pearson_corr_value = pearson_corr.values
    diagonal_indices = np.arange(min(pearson_corr_value.shape))
    pearson_corr_value[diagonal_indices,diagonal_indices] = 0
    pearson_corr_value = np.abs(pearson_corr_value)
    for row_index,row_val in enumerate(pearson_corr_value):
        # pearson_corr_value[row_index] = gui_one(pearson_corr_value[row_index])
        pearson_corr_value[row_index] = pearson_corr_value[row_index]
    return pearson_corr_value

def get_attr_map_Null_kendall(pathMiss, data_m, categorical_cols, continuous_cols):
    Miss_data = pd.read_csv(pathMiss)
    value_name = Miss_data.columns
    categorical_col, continuous_col = categorical_cols, continuous_cols
    col_num = Miss_data.shape[1]
    columns_name = Miss_data.columns.values
    data_copy = Miss_data.copy()
    columns_name_list = columns_name.tolist()
    columns_name_number = []
    columns_name_cat = []
    enc = OrdinalEncoder()
    for i in range(col_num):
        if i in continuous_col:
            columns_name_number.append(columns_name_list[i])
            Miss_data[columns_name_list[i]] = pd.cut(Miss_data[columns_name_list[i]], bins=20)
            label_encoder = LabelEncoder()
            Miss_data[columns_name_list[i]] = label_encoder.fit_transform(Miss_data[columns_name_list[i]])
        else:
            columns_name_cat.append(columns_name_list[i])
    Miss_data[columns_name_cat] = enc.fit_transform(Miss_data[columns_name_cat])

    columns_name_list = Miss_data.columns.values
    corr_map = {}
    model = DecisionTreeRegressor()

    # 计算第i列与其他每一列数据的kendall
    for index1, col_value in enumerate(columns_name_list):
        cur_corr_map = {}
        # 获取第i列所有没有缺失的数据
        no_miss_index1 = np.where(data_m[:, index1] == 1)
        for index2, col_value2 in enumerate(columns_name_list):
            no_miss_index2 = np.where(data_m[:, index2] == 1)
            common_elements = np.intersect1d(no_miss_index1, no_miss_index2)
            a = Miss_data.iloc[common_elements, :]
            b = data_copy.iloc[common_elements, :]
            index1_val = Miss_data.iloc[common_elements, :].iloc[:, index1]
            index2_val = Miss_data.iloc[common_elements, :].iloc[:, index2]
            kendall_tu = cramers_v(index1_val, index2_val)
            # kendall_tu = kendalltau(np.array(index1_val), np.array(index2_val)).correlation
            # kendall_tu = pearsonr(np.array(index1_val), np.array(index2_val)).correlation
            # kendall_tu = spearmint(np.array(index1_val), np.array(index2_val)).correlation
            cur_corr_map[col_value2] = kendall_tu
        corr_map[col_value] = cur_corr_map
    corr_map = pd.DataFrame(corr_map)
    corr_map = get_tongji_attn_map(corr_map, "Kendall", True)
    return corr_map




