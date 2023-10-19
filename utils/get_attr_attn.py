import math
import keras
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
import torch.nn.functional as F
from utils.data_loader import get_categorical_columns
from sklearn.metrics import mean_squared_error
import numpy as np
import keras.backend as K_cross
from math import sqrt
from scipy.stats.stats import kendalltau
# 将数据转为数值类型数据进行随机森林回归模型，返回模型，处理好的数据，每个属性编码后的维度
from util import Data_convert
import util
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from model.FD_model import entropy


def get_num_model(impute_data, value_num, value_cat):
    # 对数据预处理
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, value_num),
            ('cat', categorical_transformer, value_cat)
        ])
    # 获取每一列数据编码后的数据维度
    encode_num_list = []
    impute_data_processed = None
    for index in impute_data.columns.values:
        if index in value_num:
            cop = impute_data.copy()
            encode_num_list.append(1)
            encode_data = numeric_transformer.fit_transform(cop[[index]])

        else:
            cop = impute_data.copy()
            encode_data = categorical_transformer.fit_transform(cop[[index]]).toarray()
            encode_num_list.append(encode_data.shape[1])
        if impute_data_processed is None:
            impute_data_processed = encode_data
        else:
            impute_data_processed = np.hstack([impute_data_processed, encode_data])

    # 使用随机森林回归模型
    rf_num = RandomForestRegressor(n_estimators=100, max_depth=5)
    return rf_num, impute_data_processed, encode_num_list


# 将数据转为多类别数据进行随机森林分类模型，返回模型，处理好的数据，每个属性编码后的维度
def get_cat_model(impute_data, value_num, value_cat):
    # 对数据预处理
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    impute_data_processes = []
    enc = OrdinalEncoder()
    enc.fit(impute_data[value_cat])  # 学习编码规则
    impute_data_copy = impute_data.copy()
    impute_data_copy[value_cat] = enc.transform(impute_data_copy[value_cat])
    for index in impute_data.columns.values:
        if index in value_num:
            x = numeric_transformer.fit_transform(impute_data[[index]])
        else:
            x = impute_data_copy[[index]].values
        impute_data_processes.append(x)
    impute_data_processes = np.array(impute_data_processes)

    impute_data_processes = np.transpose(impute_data_processes.reshape((impute_data_processes.shape[0] , -1)))

    # criterion：衡量分裂质量的度量.
    # n_estimators：森林中树的数量。
    rf_num = RandomForestClassifier(oob_score=False, random_state=10, criterion='entropy', n_estimators=200)
    encode_num_list = None
    return rf_num, impute_data_processes,encode_num_list


def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x/e_x.sum(axis=0)

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


# 获取每列属性的缺失数量
def get_miss_num(miss_data):
    attr_miss_num = {}
    attr_name = miss_data.columns
    for attr in attr_name:
        miss_num = 0
        all_value = miss_data[attr]
        for attr_val in all_value:
            if attr_val == 0 or attr_val == "Null":
                miss_num = miss_num + 1
        attr_miss_num[attr] = miss_num
    return attr_miss_num


# 获取缺失位置的矩阵
def get_miss_M(miss_data):
    M = miss_data.copy()
    attr_name = miss_data.columns
    for index, attr in enumerate(attr_name):
        for index2, attr_val in enumerate(miss_data[attr]):
            if attr_val == 0 or attr_val == "Null":
                M.iat[index2, index] = 0
            else:
                M.iat[index2, index] = 1
    return M



def gui_one(array):
    array_sum = np.sum(array)
    res = [x/array_sum for x in array]
    return res

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

def convert_to_category(data, num_categories):
    min_val = min(data)
    max_val = max(data)
    step = (max_val - min_val) / num_categories
    categories = [min_val + i * step for i in range(num_categories)]
    result = [0] * len(data)
    for i in range(len(data)):
        for j in range(num_categories):
            if data[i] <= categories[j]:
                result[i] = j
                break
    return result
# 直接使用随机森林来获取属性间的相关性
def get_attr_map_Scikit(pathMiss, pathImpute, cat, col_num):
    miss_data = pd.read_csv(pathMiss)
    impute_data = pd.read_csv(pathImpute)
    value_name = impute_data.columns
    # corr_matrix = impute_data.corr()
    miss_num = get_miss_num(miss_data)
    categorical_cols, continuous_cols = cat, col_num
    attr_dict = {}  # 存放每个属性及其它属性对该属性的影响{{}}
    col = impute_data.shape[1]
    fields = impute_data.columns.values
    values = fields.tolist()
    value_num = []
    value_cat = []
    for i in range(col):
        if i in continuous_cols:
            value_num.append(values[i])
        else:
            value_cat.append(values[i])
    # 存储对于每个属性，其他属性对它的影响程度
    attn_corr_map = {}
    attr_corr_map_MissNum = {}
    for index in range(col):
        attn_corr_index = {}
        attn_corr_index_MissNum = {}
        if index in continuous_cols :
            model, impute_data_processed, encode_num_list = get_num_model(impute_data, value_num, value_cat)
            y_in_code = 0
            for i in range(0,index):
                y_in_code = y_in_code + encode_num_list[i]
        else:
            model, impute_data_processed, encode_num_list = get_cat_model(impute_data, value_num, value_cat)
            y_in_code = index
            # Y_train应该取index对应的那一列
        Y_train = impute_data_processed[:, y_in_code]
        if index == 0:
            X_train = impute_data_processed[:, y_in_code+1:]
        elif index == col - 1:
            X_train = impute_data_processed[:, :y_in_code]
        else:
            a = impute_data_processed[:, :y_in_code]
            b = impute_data_processed[:, y_in_code + 1:]
            X_train = np.hstack([a,b])
        X_train = np.nan_to_num(X_train)
        Y_train = np.nan_to_num(Y_train).ravel()
        model.fit(X_train, Y_train)
        # 获取其他属性对其的相关性
        importances = model.feature_importances_
        impor_index = 0
        index2_list = []
        att_list = []
        att_list_MissNum = []
        for index2 in range(col):
            if index == index2:
                continue
            if encode_num_list is None:
                att_list_MissNum.append(importances[impor_index]*(1 + math.log(2+miss_num[values[index2]])))
                att_list.append(importances[impor_index])
                impor_index = impor_index + 1
            else:
                start_col = impor_index
                end_col = start_col + encode_num_list[index2]
                importance = 0
                for impor in range(start_col, end_col):
                    importance = importance + importances[impor]
                # importance = importance/encode_num_list[index2]
                if importance < 0:
                    importance = 0
                att_list_MissNum.append(importance*(1 + math.log(2+miss_num[values[index2]])))
                att_list.append(importance)
                impor_index = end_col
            index2_list.append(index2)
        # new_loss_list = gui_one(np.array(att_list))
        # new_loss_list_MissNum = gui_one(np.array(att_list_MissNum))
        new_loss_list = np.array(att_list)
        new_loss_list_MissNum = np.array(att_list_MissNum)
        for i, index2 in enumerate(index2_list):
            attn_corr_index[values[index2]] = new_loss_list[i]
            attn_corr_index_MissNum[values[index2]] = new_loss_list_MissNum[i]
        attn_corr_map[values[index]] = attn_corr_index
        attr_corr_map_MissNum[values[index]] = attn_corr_index_MissNum
    attn_corr_map = pd.DataFrame(attn_corr_map)
    attr_corr_map_MissNum = pd.DataFrame(attr_corr_map_MissNum)
    return attn_corr_map,attr_corr_map_MissNum

def get_attr_map_tongji(pathImpute):
    impute_data = pd.read_csv(pathImpute)
    value_name = impute_data.columns
    categorical_cols, continuous_cols = get_categorical_columns(pathImpute, header=1,
                                                                categorical_num=20)  # 在这些列上是数值类型
    col = impute_data.shape[1]
    encoder = OrdinalEncoder()
    fields = impute_data.columns.values
    values = fields.tolist()
    value_num = []
    value_cat = []
    for i in range(col):
        if i in continuous_cols:
            value_num.append(values[i])
        else:
            value_cat.append(values[i])
    impute_data[value_cat] = encoder.fit_transform(impute_data[value_cat])

    # 计算 Pearson 相关系数
    person_corr = get_tongji_attn_map(impute_data, "pearson", False)

    # 计算 Spearman 相关系数
    spearman_corr = get_tongji_attn_map(impute_data, 'spearman', False)

    # 计算 Kendall 相关系数
    kendall_corr = get_tongji_attn_map(impute_data, 'kendall',False)

    # 存储对于每个属性，其他属性对它的影响程度
    return pd.DataFrame(person_corr), pd.DataFrame(spearman_corr), pd.DataFrame(kendall_corr)


# 返回训练好回归模型,编码后的数据以及每条数据编码后的的数值
def get_model_keras(impute_data, value_cat, index, continuous_cols, UserMissData, M):
    aaaa = index
    # 对impute_data进行one-hot编码，数值数据进行标准化
    impute_data_x = impute_data.copy()
    enc = OrdinalEncoder()
    enc.fit(impute_data[value_cat])
    # 将类别数据编码为1，2，3，4，5
    impute_data_x[value_cat] = enc.transform(impute_data[value_cat])
    cat_to_code_data = pd.DataFrame(impute_data_x)
    # 将1,2,3,4,5编码为one-hot
    # filed存储模型的列别  feed_data编码的数据
    cat_to_code_data.columns = [x for x in range(cat_to_code_data.shape[1])]
    fields, feed_data = Data_convert(cat_to_code_data, "meanstd", continuous_cols)
    if UserMissData:
        # 对X_train部分数据编码为0
        attr_name = M.columns
        for in1, attr in enumerate(attr_name):
            if in1 == index:
                continue
            # in1 编码后是哪些列
            start_col = 0
            for i in range(in1):
                if fields[i].dtype == 'Numerical Data':
                    start_col = start_col + 1
                else:
                    start_col = start_col + len(fields[i].dict)
            if fields[in1].dtype == 'Numerical Data':
                end_col = start_col + 1
            else:
                end_col = start_col + len(fields[in1].dict)
            for in2, attr_val in enumerate(M[attr]):
                if attr_val == 0:
                    feed_data.iloc[in2, start_col:end_col] = 0

    # 获取index列的数据作为Y
    start_col = 0
    for i in range(index):
        if fields[i].dtype == 'Numerical Data':
            start_col = start_col + 1
        else:
            start_col = start_col + len(fields[i].dict)
    if fields[index].dtype == 'Numerical Data':
        end_col = start_col + 1
    else:
        end_col = start_col + len(fields[index].dict)
    Y_train = feed_data.iloc[:, start_col:end_col]
    if index == 0:
        X_train = feed_data.iloc[:, end_col:]
    elif index == len(impute_data)-1:
        X_train = feed_data.iloc[:, :start_col]
    else:
        X_train_1 = feed_data.iloc[:, :start_col]
        X_train_2 = feed_data.iloc[:, end_col:]
        X_train = pd.concat([X_train_1,X_train_2], axis=1)

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    model = Sequential()
    model.add(Dense(input_dim/2, input_dim=input_dim, activation='relu'))
    model.add(Dense(input_dim/3, activation='relu'))
    if index in continuous_cols:
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
    else:
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 编译模型
    modelAll = model.fit(X_train.values, Y_train.values, epochs=200, batch_size=50)

    # model.save( "expdir/" + "adult" + "/" + 'adult_{}.h5'.format(aaaa))

    return model, enc, fields, feed_data, modelAll.history['loss'][-1]


# 使用填充好的数据进行
def get_attr_map_keras(pathMiss, pathImpute, AddMissNum, UserMissData):
    miss_data = pd.read_csv(pathMiss)
    M = get_miss_M(miss_data)
    impute_data = pd.read_csv(pathImpute)
    # corr_matrix = impute_data.corr()
    miss_num = get_miss_num(miss_data)
    categorical_cols, continuous_cols = get_categorical_columns(pathImpute, header=1,
                                                                categorical_num=5)  # 在这些列上是数值类型
    attr_dict = {}  # 存放每个属性及其它属性对该属性的影响{{}}
    col = impute_data.shape[1]
    fields = impute_data.columns.values
    values = fields.tolist()
    value_num = []
    value_cat = []
    for i in range(col):
        if i in continuous_cols:
            value_num.append(values[i])
        else:
            value_cat.append(values[i])
    # 存储对于每个属性，其他属性对它的影响程度
    attn_corr_map = {}
    for index in range(col):
        attn_corr_index = {}
        # 获取训练好的模型,数字编码，每列编码用的模型，编码后数据
        model, enc, fields, feed_data, ori_loss = get_model_keras(impute_data, value_cat, index, continuous_cols, UserMissData, M)
        # 获取其他属性对其的相关性
        index2_list = []
        att_list = []
        for index2 in range(col):
            if index == index2:
                continue
            # 将index2处的数据全设为0
            start_col = 0
            feed_data_copy = feed_data.copy()
            for i in range(index2):
                if fields[i].dtype == 'Numerical Data':
                    start_col = start_col + 1
                else:
                    start_col = start_col + len(fields[i].dict)
            if fields[index2].dtype == 'Numerical Data':
                end_col = start_col + 1
            else:
                end_col = start_col + len(fields[index2].dict)
            feed_data_copy.iloc[:, start_col:end_col] = 0
            start_col_index = 0
            for i in range(index):
                if fields[i].dtype == 'Numerical Data':
                    start_col_index = start_col_index + 1
                else:
                    start_col_index = start_col_index + len(fields[i].dict)
            if fields[index].dtype == 'Numerical Data':
                end_col_index = start_col_index + 1
            else:
                end_col_index = start_col_index + len(fields[index].dict)
            Y_test = feed_data_copy.iloc[:, start_col_index:end_col_index]
            if index == 0:
                X_test = feed_data_copy.iloc[:, end_col_index:]
            elif index == len(impute_data) - 1:
                X_test = feed_data_copy.iloc[:, :start_col_index]
            else:
                X_test_1 = feed_data_copy.iloc[:, :start_col_index]
                X_test_2 = feed_data_copy.iloc[:, end_col_index:]
                X_test = pd.concat([X_test_1, X_test_2], axis=1)
            Y_pred = model.predict(X_test.values)
            if index in continuous_cols:
                zero_loss = K_cross.eval(K_cross.mean(keras.losses.mean_squared_error( Y_pred, Y_test.values)))
            else:
                epsilon = 1e-9
                y_test_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
                # 获取去除一个属性后mse的差异
                zero_loss = K_cross.eval(K_cross.mean(keras.losses.categorical_crossentropy(y_test_pred, Y_test.values)))
            importance = zero_loss - ori_loss
            if AddMissNum:
                att_list.append(importance*(1 + math.log(1+miss_num[values[index2]])))
            else:
                att_list.append(importance)
            impor_index = end_col
            index2_list.append(index2)
        new_loss_list = gui_one(np.array(att_list))
        for i, index2 in enumerate(index2_list):
            attn_corr_index[values[index2]] = new_loss_list[i]
        attn_corr_map[values[index]] = attn_corr_index
    attn_corr_map2 = pd.DataFrame(attn_corr_map)
    return attn_corr_map2




# 训练好数据，通过训练好随机森林模型，然后打乱某列数据得到属性间相关性
def get_attr_map_Random(pathMiss, pathImpute):
    # path = "expdir/" + "adult" + "/"
    # miss_data = pd.read_csv(path + "miss_data_0.4.csv")
    # impute_data = pd.read_csv(path + "Imput_data_0.4.csv")
    miss_data = pd.read_csv(pathMiss)
    impute_data = pd.read_csv(pathImpute)
    # len = int(impute_data.shape[0]*0.2)
    # impute_data = impute_data.iloc[:len, :]
    #corr_matrix = impute_data.corr()
    miss_num = get_miss_num(miss_data)
    categorical_cols, continuous_cols = get_categorical_columns(pathImpute, header=1,
                                                                categorical_num=5)  # 在这些列上是数值类型
    attr_dict = {}  # 存放每个属性及其它属性对该属性的影响{{}}
    col = impute_data.shape[1]
    fields = impute_data.columns.values
    values = fields.tolist()
    value_num = []
    value_cat = []
    for i in range(col):
        if i in continuous_cols:
            value_num.append(values[i])
        else:
            value_cat.append(values[i])
    # 存储对于每个属性，其他属性对它的影响程度
    # for i in continuous_cols:
    #     data = impute_data[values[i]].values
    #     new_data = convert_to_category(data, 5)
    #     impute_data.loc[:, values[i]] = new_data
    attn_corr_map = {}
    attr_corr_map_MissNum = {}
    for index in range(col):
        if index in continuous_cols:
            model, impute_data_processed, encode_num_list = get_num_model(impute_data, value_num, value_cat)
            y_in_code = 0
            for i in range(0, index):
                y_in_code = y_in_code + encode_num_list[i]
        else:
            model, impute_data_processed, encode_num_list = get_cat_model(impute_data, value_num, value_cat)
            y_in_code = index
            # Y_train应该取index对应的那一列
        Y_train = impute_data_processed[:, y_in_code]
        if index == 0:
            X_train = impute_data_processed[:, y_in_code+1:]
        elif index == col - 1:
            X_train = impute_data_processed[:, :y_in_code]
        else:
            a = impute_data_processed[:, :y_in_code]
            b = impute_data_processed[:, y_in_code + 1:]
            X_train = np.hstack([a,b])

        X_train = np.nan_to_num(X_train)
        Y_train = np.nan_to_num(Y_train)
        model.fit(X_train, Y_train.ravel())
        print(index)
        # index为类别数据
        if encode_num_list is None:
            y_pred = model.predict_proba(X_train)
            epsilon = 1e-5
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            cro_ori = 0
            for irrow, i_Y in enumerate(Y_train):
                a = 1-y_pred[irrow, int(i_Y)]
                b = y_pred[irrow, int(i_Y)]
                cro_ori = cro_ori + np.log(b)-np.log(a)
            cro_ori = -cro_ori/len(impute_data)
        else:
            y_pred = model.predict(X_train)
            mse_ori = mean_squared_error(Y_train, y_pred)

        attn_corr_index = {}
        attn_corr_index_MissNum = {}
        # 去除某一列的数据：将某一列数据全设为0作为测试集
        loss_list = []
        loss_list_MissNum = []
        index2_list = []
        for index2 in range(col):

            if index == index2:
                continue
                # 将index2的数据变为0,impute_data_processed为最初的编码
            impute_data_processed_copy = np.copy(impute_data_processed)
            if encode_num_list is None:
                # 获取index2列数据
                colp = impute_data_processed_copy[:, index2]
                # 生成一个随机排列的索引
                idx = np.random.permutation(colp.shape[0])
                # 使用随机索引重新排列列数据
                col_shuffled = colp[idx]
                # 将打乱后的列数据重新赋值给稀疏矩阵
                impute_data_processed_copy[:, index2] = col_shuffled

            else:
                start_col = 0
                for i in range(index2):
                    start_col = start_col + encode_num_list[i]
                end_col = start_col + encode_num_list[index2]
                cols = impute_data_processed_copy[:, start_col:end_col]
                cols_as_array = cols
                idx = np.random.permutation(cols_as_array.shape[0])
                cols_shuffled = cols_as_array[idx]
                impute_data_processed_copy[:, start_col:end_col] = cols_shuffled

            if index == 0:
                X_test = impute_data_processed_copy[:, y_in_code + 1:]
            elif index == col - 1:
                X_test = impute_data_processed_copy[:, :y_in_code]
            else:
                a = impute_data_processed_copy[:, :y_in_code]
                b = impute_data_processed_copy[:, y_in_code + 1:]
                X_test = np.hstack([a, b])

            Y_test = np.copy(Y_train)
            X_test = np.nan_to_num(X_test)
            Y_test= np.nan_to_num(Y_test)

            if encode_num_list is None:
                y_test_pred = model.predict_proba(X_test)
                # Compute the mean squared error
                epsilon = 1e-5
                y_test_pred = np.clip(y_test_pred, epsilon, 1 - epsilon)
                cro_attr_zero = 0
                for irrow, i_Y in enumerate(Y_test):
                    a = 1 - y_test_pred[irrow, int(i_Y)]
                    b = y_test_pred[irrow, int(i_Y)]
                    cro_attr_zero = cro_attr_zero + np.log(b) - np.log(a)
                cro_attr_zero = -cro_attr_zero/miss_data.shape[0]
                # 获取去除一个属性后mse的差异
                loss_diff = (cro_attr_zero - cro_ori)
            else:
                y_test_pred = model.predict(X_test)
                # Compute the mean squared error
                mse_attr_zero = mean_squared_error(Y_test, y_test_pred)
                # 获取去除一个属性后mse的差异
                loss_diff = (mse_attr_zero - mse_ori)
            a = miss_num[values[index2]]
            # print(miss_data)
            b = miss_data.shape[0]
            loss_diff_MissNum = loss_diff*(1 + math.log(1 + a/b))
            loss_list.append(loss_diff)
            loss_list_MissNum.append(loss_diff_MissNum)
            index2_list.append(index2)

        new_loss_list = gui_one(np.array(loss_list))
        new_loss_list_MissNum = gui_one(np.array(loss_list_MissNum))
        for i, index2 in enumerate(index2_list):
            attn_corr_index[values[index2]] = new_loss_list[i]
            attn_corr_index_MissNum[values[index2]] = new_loss_list_MissNum[i]
        attn_corr_map[values[index]] = attn_corr_index
        attr_corr_map_MissNum[values[index]] = attn_corr_index_MissNum
    attn_corr_map = pd.DataFrame(attn_corr_map)
    attr_corr_map_MissNum = pd.DataFrame(attr_corr_map_MissNum)
    return attn_corr_map,attr_corr_map_MissNum


# 根据corr_map填充数据
def get_impute_data_map(impute_data, attr_corr_map, enc, value_cat, M):
    attr_corr_map = pd.DataFrame(attr_corr_map)
    miss_data_copy = impute_data.copy()
    miss_data_copy[value_cat] = enc.transform(miss_data_copy[value_cat])

    Dimr = miss_data_copy.shape[1]

    G_WQ = torch.tensor(util.xavier_init([Dimr, Dimr]))
    G_b1 = torch.tensor(np.zeros(shape=[Dimr]))

    G_WK = torch.tensor(util.xavier_init([Dimr, Dimr]))
    G_b2 = torch.tensor(np.zeros(shape=[Dimr]))

    G_WV = torch.tensor(util.xavier_init([Dimr, Dimr]))
    G_b3 = torch.tensor(np.zeros(shape=[Dimr]))

    inputs = torch.tensor(miss_data_copy.values).double()
    Q = F.relu(torch.matmul(inputs, G_WQ) + G_b1)
    K = F.relu(torch.matmul(inputs, G_WK) + G_b2)
    # 如果是类别的，使用sqrt(Dimr)使得Q更稳定
    V = F.relu(torch.matmul(inputs, G_WV) + G_b3)
    attn1 = F.relu(torch.mm(V, torch.softmax(torch.matmul(K.T, Q) / sqrt(Dimr), dim=1)))

    return attn1.float()


def get_tuple_sim_attrCorr(tuple_value, sim_tuple_value, categorical_cols, attr_corr_curIndex,values,data_m,ori_index,sim_index):
    num_sim = 0
    cat_sim = 0
    sim = 0
    for index in range(len(tuple_value)):
        value_name = values[index]
        corr = attr_corr_curIndex[value_name]
        if corr is None:
            corr = 0
        if data_m[ori_index, index] == 0 or data_m[sim_index, index] == 0:
            sim = sim + corr
            continue
        if index not in categorical_cols:
            num_sim = num_sim+math.sqrt((tuple_value[index]-sim_tuple_value[index])**2)* corr
        else:
            if tuple_value[index] != sim_tuple_value[index]:
                cat_sim = cat_sim + corr
    sim = sim + num_sim + cat_sim
    return sim



def get_attr_map(model_name,pathMiss, pathImpute, cat, col):
    flag = False
    if model_name == "Scikit_AddMissNum":
        attr_corr_map, _ = get_attr_map_Scikit(pathMiss, pathImpute)
    elif model_name == "Scikit_NoMissNum":
        attr_corr_map, _ = get_attr_map_Scikit(pathMiss, pathImpute, cat, col)
    elif model_name == "Random_AddMissNum":
        attr_corr_map, _ = get_attr_map_Random(pathMiss, pathImpute)
    elif model_name == "Random_NoMissNum":
        attr_corr_map, _ = get_attr_map_Random(pathMiss, pathImpute)
    elif model_name == "keras_AddMissNum":
        attr_corr_map = get_attr_map_keras(pathMiss, pathImpute, AddMissNum=True, UserMissData=False)
    elif model_name == "keras_NoMissNum":
        attr_corr_map = get_attr_map_keras(pathMiss, pathImpute, AddMissNum=False, UserMissData=False)
    elif model_name == "keras_UserAddMissData":
        attr_corr_map = get_attr_map_keras(pathMiss, pathImpute, AddMissNum=True, UserMissData=True)
    elif model_name == "keras_UserNoMissData":
        attr_corr_map = get_attr_map_keras(pathMiss, pathImpute, AddMissNum=False, UserMissData=True)
    elif model_name == "tongji":
        flag = True
        attr_corr_map_Per,Kel,Spa = get_attr_map_tongji(pathImpute)
    if not flag:
        return attr_corr_map,None,None
    else:
        return attr_corr_map_Per,Kel,Spa



def pd_corr_To_numpy_corr(corr_map):
    corr_map = corr_map[0]  # 因为返回3个map的列表
    corr_map_values = corr_map.values
    last_row = corr_map_values[-1, :]
    rest_of_array = corr_map_values[:-1, :]
    new_array = np.concatenate(([last_row], rest_of_array), axis=0)
    diagonal_indices = np.arange(min(new_array.shape))
    new_array[diagonal_indices,diagonal_indices] = 0
    pearson_corr_value = np.abs(new_array)

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


def get_attr_map_gini(miss_data, data_m, categorical_cols):
    # 获取value_cat
    value_cat = []
    # 建立一个字典
    col_dict = {}
    data = miss_data.values
    values = miss_data.columns
    # 获取所有属性的没有缺失的位置
    has_data_index = []
    for col_index, col_val in enumerate(data_m.T):
        cur_has_data_index = []
        for row_index, val in enumerate(col_val):
            if val == 1:
                cur_has_data_index.append(row_index)
        has_data_index.append(cur_has_data_index)
    # 对类别数据进行编码，数值数据分箱
    encoder_data = []
    for col_index, col_val in enumerate(data.T):
        arr = np.array(col_val)
        # 获取唯一的类别值并将其映射为整数
        if col_index in categorical_cols:
            arr = arr.astype(np.str_)
            unique_values, encoded_arr = np.unique(arr, return_inverse=True)
            encoded_arr += 1
            value_cat.append(miss_data.columns[col_index])
        else:
            # 将数据分成 5 个箱子，并为每个箱子分配一个类别编码
            bins = np.array_split(np.sort(arr), 20)
            encoded_arr = np.zeros_like(arr)
            for i, bin_ in enumerate(bins):
                encoded_arr[np.isin(arr, bin_)] = i
        col_dict[values[col_index]] = col_index
        encoder_data.append(encoded_arr)

    corr_map = {}
    # 计算第i列与其他每一列数据的信息增益
    for index1, col_value in enumerate(values):
        cur_corr_map = {}
        # 获取第i列所有没有缺失的数据
        no_miss_index1 = has_data_index[index1]
        for index2, col_value2 in enumerate(values):
            newShang = 0
            no_miss_index2 = has_data_index[index2]
            common_elements = np.intersect1d(no_miss_index1, no_miss_index2)
            label_data = encoder_data[index1][common_elements]
            ori_shang = entropy(label_data)
            feature_data = encoder_data[index2][common_elements]
            # 对每个值划分数据集，newShang为i列数据获取的熵值
            unique_values, counts = np.unique(feature_data, return_counts=True)
            probabilities = counts / len(feature_data)
            for val, probability in zip(unique_values, probabilities):
                subset_labels = label_data[feature_data == val]
                newShang += probability * entropy(subset_labels)
            infoGain = ori_shang - newShang
            cur_corr_map[col_value2] = infoGain
        corr_map[col_value] = cur_corr_map
    corr_map = pd.DataFrame(corr_map)
    # 归一化
    corr_map = get_tongji_attn_map(corr_map, "Kendall", True)
    return corr_map



if __name__ == "__main__":
    AddMissNum = True
    UseMissData = False
    path = "expdir/" + "adult" + "/"
    pathMiss = path + "miss_data_0.5.csv"
    # pathMiss = path + "adult.csv"
    # pathImpute = path + "Imput_miss_data_0.4.csv"
    pathImpute = path + "Imput_miss_data_0.5.csv"
    pathTrue = path + "true_data_0.2.csv"
    data_M = pd.read_csv(path + "M_0.5.csv")
    # attr_corr_map_Random, attr_corr_map_Random_MissNum = get_attr_map_Random(pathMiss, pathImpute)
    # attr_corr_map_Random.to_csv(path+"Random.csv")
    # attr_corr_map_Random_MissNum.to_csv(path+"Random_MissNum.csv")
    pathOri = path + "ori_data.csv"
    ori_data = pd.read_csv(pathOri)
    # 获取统一的编码方式
    col = ori_data.shape[1]
    fields = ori_data.columns.values
    values = fields.tolist()
    value_num = []
    value_cat = []
    categorical_cols, continuous_cols = get_categorical_columns(pathOri, header=1,
                                                                categorical_num=20)
    for i in range(col):
        if i in continuous_cols:
            value_num.append(values[i])
        else:
            value_cat.append(values[i])
    copy_ori_data = ori_data.copy()
    enc = OrdinalEncoder()
    enc.fit(copy_ori_data[value_cat])
    # 获取M
    miss_data = pd.read_csv(pathMiss)
    M = data_M.values
    # K_NN填充
    # Impute_data_KNN = pd.read_csv(pathImpute)
    # Loss_NN = utils.errorLoss(Impute_data_KNN, ori_data, M, value_cat, continuous_cols, enc)
    # print("NN_Loss      {}\n".format(Loss_NN))
    # log = open(path + "attr_corr_Loss.txt", "a+")
    # log.write("NN_Loss      {}\n".format(Loss_NN))
    # log.close()


    # Pearson,Spearman,Kendall
    person_corr, sperrman_corr, kendall_corr = get_attr_map("tongji", pathMiss, pathTrue)

    # Scikit_NoMissNum填充
    Scikit_corr = get_attr_map("Scikit_NoMissNum", pathMiss, pathTrue)
    Scikit_corr = pd_corr_To_numpy_corr(Scikit_corr)
    # Impute_data_KNN = pd.read_csv(path+"Scikit_NoMissNum_imput.csv")
    # Loss_NN = utils.errorLoss(Impute_data_KNN, ori_data, M, value_cat, continuous_cols, enc)
    # print("Scikit_NoMissNum_Loss      {}\n".format(Loss_NN))
    # log = open(path + "attr_corr_Loss.txt", "a+")
    # log.write("Scikit_NoMissNum_Loss      {}\n".format(Loss_NN))
    # log.close()


    Random_corr = get_attr_map("Random_NoMissNum", pathMiss, pathTrue)
    Random_corr = pd_corr_To_numpy_corr(Random_corr)
    # Impute_data_KNN = pd.read_csv(path+"Random_NoMissNum_imput.csv")
    # Loss_NN = utils.errorLoss(Impute_data_KNN, ori_data, M, value_cat, continuous_cols, enc)
    # print("Random_NoMissNum_Loss      {}\n".format(Loss_NN))
    # log = open(path + "attr_corr_Loss.txt", "a+")
    # log.write("Random_NoMissNum_Loss      {}\n".format(Loss_NN))
    # log.close()
    #
    Keras_corr = get_attr_map("keras_NoMissNum", pathMiss, pathTrue)
    Keras_corr = pd_corr_To_numpy_corr(Keras_corr)
    # Impute_data_KNN = pd.read_csv(path+"keras_NoMissNum_imput.csv")
    # Loss_NN = utils.errorLoss(Impute_data_KNN, ori_data, M, value_cat, continuous_cols, enc)
    # print("keras_NoMissNum_Loss      {}\n".format(Loss_NN))
    # log = open(path + "attr_corr_Loss.txt", "a+")
    # log.write("keras_NoMissNum_Loss      {}\n".format(Loss_NN))
    # log.close()


