# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from util import reconvert_data


class FDModel(nn.Module):
    def __init__(self, input_size, output_size, x_index_list, y_index):
        super(FDModel, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.x_index_list = x_index_list
        self.y_index = y_index

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def set(self, new_x_index_list):
        self.x_index_list = new_x_index_list



def train_Model(x, y, model):

    train_dataset = TensorDataset(x, y)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    for epoch in range(300):
        for batch_features, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward(retain_graph=True)
            optimizer.step()
        if epoch % 100 == 0:
            acc = test_model(train_dataloader, model)
            if acc == 1:
                break
    accuracy = test_model(train_dataloader, model)
    return model


def test_model(data, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in data:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    accuracy = correct / total
    return accuracy

def entropy(labels):
    if len(labels) == 0:
        return 0
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def get_my_FD_loss(FD_model_list, decode_code, fields):
    if len(FD_model_list) == 0:
        return 0
    cur_dim = 0
    col_dim = [0]
    all_loss = 0
    for index, field in enumerate(fields):
        if field.data_type == 'Categorical Data':
            dim = len(field.dict)
        else:
            dim = 1
        cur_dim += dim
        col_dim.append(cur_dim)
    criterion = nn.CrossEntropyLoss()
    mse = nn.CrossEntropyLoss()
    for FD_model in FD_model_list:

        x_data_index = FD_model.x_index_list
        y_index = FD_model.y_index
        x_list = []
        for x_index in range(len(fields)):
            if x_index in x_data_index:
                x_code = decode_code[:, col_dim[x_index]:col_dim[x_index + 1]]
                x_list.append(x_code)
            elif x_index != y_index:
                x_code = torch.zeros((decode_code.shape[0], col_dim[x_index + 1] - col_dim[x_index]), dtype=decode_code.dtype).to(decode_code.device)
                x_list.append(x_code)
        x = torch.cat(x_list, dim=1)
        y_code = decode_code[:, col_dim[y_index]:col_dim[y_index + 1]]
        y = torch.argmax(y_code, dim=1).long()
        with torch.no_grad():
            outputs = FD_model(x)
            loss = criterion(outputs, y)
            # loss = mse(outputs, y_code)
            all_loss += loss
    return all_loss

def get_FD_model_Tree(miss_data, data_m, categorical_cols, zero_feed_data, fields, device, sort_corr_dict):
    value_cat = []
    col_dict = {}
    data = miss_data.values
    values = miss_data.columns
    has_data_index = []
    observer_data = []
    for col_index, col_val in enumerate(data_m.T):
        cur_has_data_index = []
        cur_observer_data = []
        for row_index, val in enumerate(col_val):
            if val == 1:
                cur_has_data_index.append(row_index)
                cur_observer_data.append(data[row_index][col_index])
        has_data_index.append(cur_has_data_index)
        observer_data.append(cur_observer_data)
    encoder_data = []
    new_sort_corr_dict = {}
    for col_index, col_val in enumerate(data.T):
        arr = np.array(col_val)
        cur_corr_sort = []
        if col_index in categorical_cols:
            arr = arr.astype(np.str_)
            unique_values, encoded_arr = np.unique(arr, return_inverse=True)
            encoded_arr += 1
            value_cat.append(miss_data.columns[col_index])
        else:
            # bins = np.linspace(arr.min(), arr.max(), 10)
            # encoded_arr = np.digitize(arr, bins)
            bins = np.array_split(np.sort(arr), 10)
            encoded_arr = np.zeros_like(arr)
            for i, bin_ in enumerate(bins):
                encoded_arr[np.isin(arr, bin_)] = i
        col_dict[miss_data.columns[col_index]] = col_index
        col_index_corr_sort = sort_corr_dict[col_index]
        for i in col_index_corr_sort:
            if i in categorical_cols and i != col_index:
                cur_corr_sort.append(i)
        new_sort_corr_dict[col_index] = cur_corr_sort
        encoder_data.append(encoded_arr)
    a = np.array(encoder_data)
    col_tree = {}
    for col_name in value_cat:
        if col_name == "state":
            print(1)
        featLabels = []
        choose_row_index = np.array(has_data_index[col_dict[col_name]])
        graph = {}
        for node in new_sort_corr_dict[col_dict[col_name]]:
            graph[node] = new_sort_corr_dict[col_dict[col_name]][new_sort_corr_dict[col_dict[col_name]].index(node) + 1:]
        buildTreeDFS(encoder_data, col_dict, col_name, has_data_index, featLabels, choose_row_index, graph)
        col_tree[col_name] = featLabels

    models = []
    for key, value in col_tree.items():
        if len(value) > 0:
            y_index = col_dict[key]
            # x_index_list = get_x_index(value)
            x_index_list = value
            for x_index in x_index_list:
                model = get_model_by_tree(x_index, y_index, zero_feed_data, has_data_index, fields, device)
                if model is not None:
                    models.append(model)
    return models


def get_model_by_tree(x_index, y_index, zero_feed_data, has_data_index, fields, device):
    set_index = set(has_data_index[y_index])
    for x in x_index:
        set_index = set_index.intersection(set(has_data_index[x]))
    set_index = np.array(list(set_index))
    X = []
    begin_list = [0]
    begin = 0
    for index,field in enumerate(fields):
        if field.data_type == "Categorical Data":
            begin += len(field.dict)
        else:
            begin += 1
        begin_list.append(begin)
    for x in range(len(fields)):
        if x in x_index:
            cur_data = zero_feed_data[set_index]
            cur_data = cur_data[:, begin_list[x]:begin_list[x+1]]
            X.append(cur_data)
        elif x != y_index:
            cur_data = torch.zeros((len(set_index), begin_list[x+1] - begin_list[x]), dtype=zero_feed_data.dtype).to(device)
            X.append(cur_data)
    X = torch.cat(X, dim=1).to(device)
    Y = zero_feed_data[set_index]
    Y = Y[:, begin_list[y_index]:begin_list[y_index+1]]
    Y = torch.argmax(Y, dim=1).long().to(device)
    input_dim = X.shape[1]
    output_dim = begin_list[y_index+1] - begin_list[y_index]
    model = FDModel(input_dim, output_dim, x_index, y_index).to(device)
    model = train_Model(X, Y, model)
    return model


def get_x_index(nested_list):
    result_list = []
    for item in nested_list:
        if isinstance(item, list):
            if len(result_list) == 0:
                for sub_item in item:
                    result_list.append([sub_item])
            else:
                sub_list = get_x_index(item)
                res_list = result_list.copy()
                cur_list = res_list.copy()
                result_list = []
                for sub_item in sub_list:
                    cur_list.append(sub_item)
                    result_list.append(cur_list)
                    cur_list = res_list.copy()
        else:
            result_list.append(item)
    return result_list

def buildTreeDFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph):
    for node in graph.keys():
        cur_node = []
        DFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph, cur_node, node)

def DFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph, cur_node, start_node):
    # if col_dict[col_name] == 1:
    #     print(1)
    cur_node.append(start_node)
    if isCurNodeInFDs(cur_node, featLabels):
        return
    if len(cur_choose_row_index) == 0:
        return
    if len(cur_node) == 3:
        return
    if isCurNodeFD(encoder_data, col_dict, col_name, has_data_index, cur_node):
        new_node = cur_node.copy()
        updateFeatLabels(new_node, featLabels)
        return
    for node in graph[start_node]:
        if node not in cur_node:
            start_node = node
            DFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph,
                         cur_node, start_node)

            cur_node.pop()

def updateFeatLabels(new_node, featLabels):
    #判断featLabels中是否有它的子集
    node_list = []
    for node in featLabels:
        if set(new_node) <= set(node) and len(new_node) > 0:
            node_list.append(node)
    if len(node_list) > 0:
        for n in node_list:
            featLabels.remove(n)
    featLabels.append(new_node)

def isCurNodeInFDs(cur_node, featLabels):
    if len(featLabels) == 0 or len(cur_node) == 0:
        return False
    for node in featLabels:
        if set(node) <= set(cur_node):
            return True
    return False


def isCurNodeFD(encoder_data, col_dict, col_name, has_data_index, cur_node):
    col_index = col_dict[col_name]
    right_observe_index = has_data_index[col_index]
    observe_index = right_observe_index
    if len(cur_node) == 0:
        return False
    for node in cur_node:
        left_observe_index = has_data_index[node]
        observe_index = list(set(left_observe_index).intersection(set(observe_index)))
    if len(observe_index) == 0:
        return False
    newShang = 0
    label_data = encoder_data[col_index][observe_index]
    feature_data = []
    for node in cur_node:
        feature_data.append(encoder_data[node][observe_index])
    feature_data = np.transpose(np.array(feature_data))
    unique_values, counts = np.unique(feature_data, axis=0, return_counts=True)
    probabilities = counts / len(feature_data)
    for value, probability in zip(unique_values, probabilities):
        a = np.where(np.all(feature_data == value, axis=1))
        subset_labels = label_data[np.where(np.all(feature_data == value, axis=1))[0]]
        newShang += probability * entropy(subset_labels)
        if newShang > 0:
            return False
    if newShang == 0:
        return True
    else:
        return False


def train_new_FD_model(new_X_index_list, y_index, un_satisfy_tuples, FD, new_x_code, fields, device):
    no, dim = new_x_code.shape
    all_index = set([i for i in range(no)])
    un_satisfy_tuples_index = set()
    for sub_set in un_satisfy_tuples:
        un_satisfy_tuples_index.update(sub_set)
    length = len(list(all_index - un_satisfy_tuples_index))
    set_index = list(all_index - un_satisfy_tuples_index)
    X = []
    begin_list = [0]
    begin = 0
    for index, field in enumerate(fields):
        if field.data_type == "Categorical Data":
            begin += len(field.dict)
        else:
            begin += 1
        begin_list.append(begin)
    for x in range(len(fields)):
        if x in new_X_index_list:
            cur_data = new_x_code[set_index]
            cur_data = cur_data[:, begin_list[x]:begin_list[x + 1]]
            X.append(cur_data)
        elif x != y_index:
            cur_data = torch.zeros((length, begin_list[x + 1] - begin_list[x])).to(device)
            X.append(cur_data)
    X = torch.cat(X, dim=1).to(device)
    Y = new_x_code[set_index]
    Y = Y[:, begin_list[y_index]:begin_list[y_index + 1]]
    Y = torch.argmax(Y, dim=1).long().to(device)
    input_dim = X.shape[1]
    output_dim = begin_list[y_index + 1] - begin_list[y_index]

    model = train_Model(X, Y, FD)
    return model


def get_trueProObserve(generate_x,decoder_z_impute,fields,data_m,device):
    truePro = torch.zeros(generate_x.shape[0], len(fields)).to(device)
    cur_index = 0
    for index, field in enumerate(fields):
        if field.data_type == "Categorical Data":
            dim = field.dim()
            data = generate_x[:, cur_index:cur_index + dim]
            zero_data = decoder_z_impute[:, cur_index:cur_index + dim]
            _, max_data_index = torch.max(data, dim=1, keepdim=True)
            _, max_zero_data_index = torch.max(zero_data, dim=1, keepdim=True)
            truePro[:, index] = torch.where(max_data_index == max_zero_data_index, torch.tensor(1).to(device),torch.tensor(0).to(device)).squeeze(-1)
            cur_index = cur_index + dim
        else:
            cur_index = cur_index + 1
    truePro = truePro.cpu().numpy()
    truePro = truePro * data_m
    return truePro



def get_eq_dict(values, miss_data_x, data_m):
    all_val_eq_dict = {}
    for attr_index,attr_name in enumerate(values):
        cur_dict = {}
        attr_data = miss_data_x.iloc[:,attr_index]
        for tup_index, val in enumerate(attr_data):
            if data_m[tup_index, attr_index] == 1:
                if val not in cur_dict.keys():
                    cur_dict[val] = [tup_index]
                else:
                    cur_dict[val].append(tup_index)
        all_val_eq_dict[attr_index] = cur_dict
    return all_val_eq_dict


# flag 0-----len(new_x_index_list)-1
def di_gui_get_sublist(split_subset, new_x_index_list, new_eq_dict, flag, impute_data):
    if flag == len(new_x_index_list):
        return
    new_split_sublist = []
    for sublist_begin in split_subset:
        if len(sublist_begin) == 1:
            continue
        cur_split_sublist = []
        other_attr = new_x_index_list[flag]
        other_attr_dict = new_eq_dict[other_attr]
        for element in sublist_begin:
            if not any(element in split for split in cur_split_sublist):
                other_attr_val = impute_data.iloc[element, other_attr]
                other_attr_sublist = other_attr_dict[other_attr_val]
                intersection_element = list(set(sublist_begin).intersection(set(other_attr_sublist)))
                new_split_sublist.append(intersection_element)
                cur_split_sublist.append(intersection_element)
    flag = flag + 1
    di_gui_get_sublist(new_split_sublist, new_x_index_list, new_eq_dict, flag, impute_data)
    return new_split_sublist


def get_satisfy_unsatisfy_tuples(RES_X, new_eq_dict, y_index, impute_data, tuple_acc_list):
    satisfy_tuples = []
    un_satisfy_tuples = []
    for sublist in RES_X:
        if len(sublist) == 1:
            continue
        y_attr_dict = new_eq_dict[y_index]
        other_attr_val = impute_data.iloc[sublist[0], y_index]
        other_attr_sublist = y_attr_dict[other_attr_val]
        intersection_element = list(set(sublist).intersection(set(other_attr_sublist)))
        if len(intersection_element) == len(sublist):
            satisfy_tuples.append(intersection_element)
            un_satisfy_tuples.append([])
        else:
            split_tuples = []
            for element in sublist:
                if not any(element in split for split in split_tuples):
                    y_attr_val = impute_data.iloc[element, y_index]
                    y_attr_sublist = y_attr_dict[y_attr_val]
                    intersection_element = list(set(sublist).intersection(set(y_attr_sublist)))
                    split_tuples.append(intersection_element)

            score_true_tup = []
            score_true_max = 0
            max_tup_score = 0
            for each_split_tuples in split_tuples:
                cur_score = 0
                cur_max_score = 0
                for tuple in each_split_tuples:
                    cur_score = cur_score + tuple_acc_list[tuple]
                    if tuple_acc_list[tuple] > cur_max_score:
                        cur_max_score = tuple_acc_list[tuple]
                if cur_max_score > max_tup_score:
                    max_tup_score = cur_max_score
                    score_true_tup = each_split_tuples
                # if cur_score > score_true_max:
                #     score_true_max = cur_score
                #     score_true_tup = each_split_tuples
            satisfy_tuples.append(score_true_tup)
            un_satisfy_tuples.append(list(set(sublist) - set(score_true_tup)))
    return satisfy_tuples, un_satisfy_tuples


def get_FD_score(satisfy_tuples, un_satisfy_tuples, tuple_acc_list):
    satisfy_score = 0
    satisfy_tuple_num = 0
    for each_satisfy_tuples in satisfy_tuples:
        satisfy_tuple_num = satisfy_tuple_num + len(each_satisfy_tuples)
        for each_tuple in each_satisfy_tuples:
            satisfy_score = satisfy_score + tuple_acc_list[each_tuple]
    un_satisfy_score = 0
    un_satisfy_tup_num = 0
    max_un_sa_tup_score = 0
    all_satisfy_tup_inS = []
    all_satisfy_tup_inS_acc = []
    un_satisfy_tup = ''
    for un_satisfy_tuples_index,each_un_satisfy_tuples in enumerate(un_satisfy_tuples):
        un_satisfy_tup_num = un_satisfy_tup_num + len(each_un_satisfy_tuples)
        for each_tuple in each_un_satisfy_tuples:
            un_satisfy_score = un_satisfy_score + tuple_acc_list[each_tuple]
            if tuple_acc_list[each_tuple] > max_un_sa_tup_score:
                all_satisfy_tup_inS = []
                all_satisfy_tup_inS_acc = []
                un_satisfy_tup = ''
                max_un_sa_tup_score = tuple_acc_list[each_tuple]
                un_satisfy_tup = each_tuple
                all_satisfy_tup_inS = satisfy_tuples[un_satisfy_tuples_index]
                for tup in all_satisfy_tup_inS:
                    all_satisfy_tup_inS_acc.append(tuple_acc_list[tup])
    if un_satisfy_tup_num == 0:
        un_satisfy_tup_num = un_satisfy_tup_num + 1
    un_satisfy_score_avg = un_satisfy_score / un_satisfy_tup_num
    # m = (1 - miss_rate) / miss_rate
    if satisfy_score + un_satisfy_score == 0:
        un_satisfy_score = 1
    return satisfy_score / (satisfy_score + un_satisfy_score) , max_un_sa_tup_score, un_satisfy_tup, all_satisfy_tup_inS, all_satisfy_tup_inS_acc

def di_gui_add_attr(x_index_list, y_index, tuple_acc_list, new_eq_dict, split_subset, impute_data, x_list, continuous_cols):
    if len(x_index_list) > 3:
        return [], [], []
    choose_index = -1
    max_score = -10000
    max_FD_score = -10000
    cur_satisfy_tuples = []
    for attr_index in new_eq_dict.keys():
        cur_x_index_list = x_index_list.copy()
        if attr_index in x_index_list or attr_index == y_index or attr_index in continuous_cols:
            continue
        cur_x_index_list.append(attr_index)
        flag = 0
        for cur_x in x_list:
            if set(cur_x).issubset(set(cur_x_index_list)):
                flag = 1
                break
        if flag == 1:
            continue
        new_split_sublist = []
        attr_dict = new_eq_dict[attr_index]
        for sublist_begin in split_subset:
            if len(sublist_begin) == 1:
                continue
            cur_split_sublist = []
            for element in sublist_begin:
                if not any(element in split for split in cur_split_sublist):
                    other_attr_val = impute_data.iloc[element, attr_index]
                    other_attr_sublist = attr_dict[other_attr_val]
                    intersection_element = list(set(sublist_begin).intersection(set(other_attr_sublist)))
                    new_split_sublist.append(intersection_element)
                    cur_split_sublist.append(intersection_element)
        satisfy_tuples, un_satisfy_tuples = get_satisfy_unsatisfy_tuples(new_split_sublist, new_eq_dict, y_index,impute_data, tuple_acc_list)
        FD_score, un_score, un_satisfy_tup, all_satisfy_tup_inS, all_satisfy_tup_inS_acc = get_FD_score(satisfy_tuples, un_satisfy_tuples, tuple_acc_list)
        new_FD_score = FD_score - un_score
        if max_score < new_FD_score:
            max_FD_score = FD_score
            max_score = new_FD_score
            choose_index = attr_index
            cur_satisfy_tuples = un_satisfy_tuples
    x_index_list.append(choose_index)
    if max_score > 0:
        return x_index_list, cur_satisfy_tuples, max_FD_score
    return di_gui_add_attr(x_index_list, y_index, tuple_acc_list, new_eq_dict, split_subset, impute_data, x_list, continuous_cols)



def update_FD_models(generate_x, zero_feed_data, fields, data_m, M_tensor, value_cat, values, miss_data_x, enc, device, eq_dict, FDs_model_list, cell_acc, continuous_cols, cost_time):
    x_code = generate_x * (1 - M_tensor) + M_tensor * zero_feed_data
    new_x_code = x_code.clone()
    impute_data = reconvert_data(x_code, fields, value_cat, values, miss_data_x, data_m, enc)
    impute_data = pd.DataFrame(impute_data)
    impute_data.columns = values
    numpy_impute_data = impute_data.values
    all_index = []
    for i in range(impute_data.shape[1]):
        if i not in continuous_cols:
            all_index.append(i)

    new_eq_dict = eq_dict.copy()
    reversed_dict = {}
    for attr_index, attr_name in enumerate(values):
        attr_data = impute_data.iloc[:, attr_index]
        for tup_index, val in enumerate(attr_data):
            if data_m[tup_index, attr_index] == 0:
                if val not in new_eq_dict[attr_index].keys():
                    new_eq_dict[attr_index][val] = [tup_index]
                else:
                    new_eq_dict[attr_index][val].append(tup_index)

    new_FD_model_list = FDs_model_list.copy()
    for FD in FDs_model_list:
        x_index_list = FD.x_index_list
        y_index = FD.y_index

        x_list = []
        for cur_FD in new_FD_model_list:
            if cur_FD.y_index == y_index and cur_FD.x_index_list != x_index_list:
                x_list.append(cur_FD.x_index_list)
        index_list = x_index_list.copy()
        index_list.append(y_index)

        tuple_acc_list = []
        for tuple_index, tuple_acc in enumerate(cell_acc):
            tuple_acc_list.append(min(tuple_acc[index_list]))


        begin_index = 0
        cur_len = 10000000
        for cur_x_index in x_index_list:
            if len(new_eq_dict[cur_x_index]) < cur_len:
                begin_index = cur_x_index
                cur_len = len(new_eq_dict[cur_x_index])

        new_x_index_list = x_index_list.copy()
        new_x_index_list.remove(begin_index)
        begin_index_val_dict = new_eq_dict[begin_index]
        if len(new_x_index_list) > 0:
            RES_X = di_gui_get_sublist(begin_index_val_dict.values(), new_x_index_list, new_eq_dict, 0, impute_data)
        else:
            RES_X = begin_index_val_dict.values()
        satisfy_tuples, un_satisfy_tuples = get_satisfy_unsatisfy_tuples(RES_X, new_eq_dict, y_index, impute_data,tuple_acc_list)
        a = x_index_list
        b = y_index

        # if y_index == 5:
        #     print(1)
        FD_score, un_sat_score, un_satisfy_tup, all_satisfy_tup_inS, all_satisfy_tup_inS_acc = get_FD_score(satisfy_tuples, un_satisfy_tuples, tuple_acc_list)  # [0, 1],0表示均不满足,1表示均满足
        # FD_score = math.log(satisfy_score) - un_satisfy_score
        if len(FD.x_index_list) == 1 and FD.x_index_list[0] == 0 and FD.y_index == 2:
            print('FD score：{}, against FD tuples：{}, against FD tuples score：{}，meet FD tuples：{}，meet FD tuples score：{}'.format(FD_score,numpy_impute_data[un_satisfy_tup][all_index], un_sat_score,numpy_impute_data[all_satisfy_tup_inS][all_index], all_satisfy_tup_inS_acc))
        old_FD_score = FD_score
        if FD_score-un_sat_score < 0:
            filename = 'result.txt'
            print('remove FD model_label_index:{},    model_feature_index:{}'.format(FD.y_index, FD.x_index_list))
            print('FD score：{}, against FD tuples：{}, against FD tuples score：{}，meet FD tuples：{}，meet FD tuples score：{}'.format(FD_score,numpy_impute_data[un_satisfy_tup][all_index], un_sat_score,numpy_impute_data[all_satisfy_tup_inS][all_index], all_satisfy_tup_inS_acc))
            with open(filename, 'w') as file:
                print('remove FD model_label_index:{},    model_feature_index:{}'.format(FD.y_index, FD.x_index_list), file = file)
                print('FD score：{}, against FD tuples：{},  against FD tuples score：{}，meet FD tuples：{}，meet FD tuples score：{}'.format(FD_score,numpy_impute_data[un_satisfy_tup], un_sat_score,numpy_impute_data[all_satisfy_tup_inS], all_satisfy_tup_inS_acc),file=file)

            if len(x_index_list) >= 2:
                new_FD_model_list.remove(FD)
            else:
                new_X_index_list, satisfy_tuples, max_FD_score = di_gui_add_attr(x_index_list, y_index, tuple_acc_list, new_eq_dict, RES_X, impute_data, x_list, continuous_cols)
                if len(new_X_index_list) > 0:
                    print('refined FD：{}--->{}，FD score：{}'.format(new_X_index_list,FD.y_index,max_FD_score))
                    with open(filename, 'w') as file:
                        print('refined FD：{}--->{}，FD score：{}'.format(new_X_index_list, FD.y_index, max_FD_score),file=file)
                    FD.set(new_X_index_list)
                    start_time = time.time()
                    new_FD_model = train_new_FD_model(new_X_index_list, y_index, un_satisfy_tuples, FD, new_x_code, fields, device)
                    # new_FD_model = train_new_FD_model_only_x(new_X_index_list, y_index, un_satisfy_tuples,new_x_code,
                    #                                   fields, device)
                    end_time = time.time()
                    duration = end_time - start_time
                    cost_time += duration
                    new_FD_model_list.append(new_FD_model)
                new_FD_model_list.remove(FD)
    return new_FD_model_list
