import numpy as np
#获取学习器的损失
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import torch

def get_L_loss(imputed_data,test_data,label_data,enc,value_cat,random_integers):
    copy_imput_data = imputed_data
    copy_test_data = test_data

    #根据imputed_data，,label_data训练AdaBoostClassifier
    copy_imput_data[value_cat] = enc.transform(copy_imput_data[value_cat])
    copy_test_data[value_cat] = enc.transform(copy_test_data[value_cat])
    #对数据进行归一化
    for index, name in enumerate(copy_imput_data.columns):
        min_val = copy_imput_data[name].min()
        max_val = copy_imput_data[name].max()
        def normalize(x):
            return (x - min_val) / (max_val - min_val) * 2 - 1
        copy_imput_data[name] =copy_imput_data[name].apply(normalize)

    for index, name in enumerate(copy_test_data.columns):
        min_val = copy_test_data[name].min()
        max_val = copy_test_data[name].max()
        def normalize(x):
            return (x - min_val) / (max_val - min_val) * 2 - 1
        copy_test_data[name] =copy_test_data[name].apply(normalize)
    X_train = np.array(copy_imput_data.values)
    Y_train = np.array(label_data.values)
    X_test =  np.array(copy_test_data.values)
    Y_test = []
    for index in random_integers:
        Y_test.append(list(label_data.values)[index])
    Y_test = np.array(Y_test)
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, Y_train)

    pre = clf.predict(X_test)
    loss = log_loss(Y_test,pre)
    loss = torch.tensor(loss)
    return loss

def RF_loss(imputed_data,test_data,label_data,enc,value_cat,random_integers):
    copy_imput_data = imputed_data
    copy_test_data = test_data
    # 根据imputed_data，,label_data训练AdaBoostClassifier
    copy_imput_data[value_cat] = enc.transform(copy_imput_data[value_cat])
    copy_test_data[value_cat] = enc.transform(copy_test_data[value_cat])
    # 对数据进行归一化
    for index, name in enumerate(copy_imput_data.columns):
        min_val = copy_imput_data[name].min()
        max_val = copy_imput_data[name].max()

        def normalize(x):
            return (x - min_val) / (max_val - min_val) * 2 - 1

        copy_imput_data[name] = copy_imput_data[name].apply(normalize)

    for index, name in enumerate(copy_test_data.columns):
        min_val = copy_test_data[name].min()
        max_val = copy_test_data[name].max()

        def normalize(x):
            return (x - min_val) / (max_val - min_val) * 2 - 1

        copy_test_data[name] = copy_test_data[name].apply(normalize)
    X_train = np.array(copy_imput_data.values)
    Y_train = np.array(label_data.values)
    X_test = np.array(copy_test_data.values)
    Y_test = []
    for index in random_integers:
        Y_test.append(list(label_data.values)[index])
    Y_test = np.array(Y_test)
    rf = RandomForestClassifier(oob_score=False, random_state=10, criterion='entropy', n_estimators=162, max_features=3,
                                max_depth=11)
    rf.fit(X_train, Y_train)
    # Extract the parameters from the model
    params = rf.get_params()

    # Convert the parameters to PyTorch tensors
    for key, value in params.items():
        params[key] = torch.tensor(value)

    Y_predictions1 = rf.predict(X_test)

    Y_predictions1 = Y_predictions1.reshape(-1, 1)
    Y_test1 = Y_test
    Y_test1 = Y_test1.reshape(-1, 1)

    lable = [0, 1]
    lable = np.array(lable, 'float32')

    G_LossValid = log_loss(Y_predictions1, Y_test1, labels=lable)
    return torch.tensor(G_LossValid).double()

