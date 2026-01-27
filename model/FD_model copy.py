# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils.util import reconvert_data
from tqdm import tqdm

# Configurable FD left-hand-side attribute limit (default 3)
FD_TAU = 3

def set_fd_tau(tau):
    """Set the FD left-hand-side attribute limit (tau)."""
    global FD_TAU
    try:
        FD_TAU = int(tau)
    except Exception:
        FD_TAU = 3


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


def get_FD_model_Tree_GroundTruth(miss_data, data_m, categorical_cols, zero_feed_data, fields, device, sort_corr_dict,data_name):
    # 获取value_cat
    value_cat = []
    # 建立一个字典
    col_dict = {}
    data = miss_data.values
    values = miss_data.columns
    # 获取所有属性的没有缺失的位置
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
    # 对类别数据进行编码，数值数据分箱
    encoder_data = []
    # 只看类别属性的相关度
    new_sort_corr_dict = {}
    for col_index, col_val in enumerate(data.T):
        arr = np.array(col_val)
        cur_corr_sort = []
        # 获取唯一的类别值并将其映射为整数
        if col_index in categorical_cols:
            arr = arr.astype(np.str_)
            unique_values, encoded_arr = np.unique(arr, return_inverse=True)
            encoded_arr += 1
            value_cat.append(miss_data.columns[col_index])
        else:
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
    # 对每个类别数据，建立决策树
    col_tree = {}
    if data_name == 'adult':
        for index,col_name in enumerate(value_cat):
            featLabels = []
            if index == 3:
                featLabels.append([4])
            elif index == 4:
                featLabels.append([3])
            col_tree[col_name] = featLabels
    elif data_name == 'hospital':
        for index,col_name in enumerate(value_cat):
            featLabels = []
            if index == 1:
                featLabels.append([0])
            elif index == 2:
                featLabels.append([0])
                featLabels.append([1])
            elif index == 3:
                featLabels.append([0])
            elif index == 4:
                featLabels.append([5])
                featLabels.append([6])
            elif index == 5:
                featLabels.append([6])
            elif index == 6:
                featLabels.append([5])
            col_tree[col_name] = featLabels
    else:
        for index,col_name in enumerate(value_cat):
            featLabels = []
            if index == 2:
                featLabels.append([0])
            elif index == 5:
                featLabels.append([2,3])
                featLabels.append([0,3])
                featLabels.append([2,6])
                featLabels.append([0,6])
            elif index == 6:
                featLabels.append([2,3])
                featLabels.append([0,3])
                featLabels.append([2,5])
                featLabels.append([0,5])
            col_tree[col_name] = featLabels
    models = []
    FD_list = []
    for key, value in col_tree.items():
        if len(value) > 0:
            y_index = col_dict[key]
            # x_index_list = get_x_index(value)
            x_index_list = value
            for x_index in x_index_list:
                new_x_index = [miss_data.columns[i] for i in x_index]
                FD_list.append('{}---->{}'.format(new_x_index, miss_data.columns[y_index]))
                # 传入x_index, y_index, zero_feed_data, has_data_index, fields  输出model
                model = get_model_by_tree(x_index, y_index, zero_feed_data, has_data_index, fields, device)
                if model is not None:
                    models.append(model)
    print("所有的FD数量为：{}\n具体的FD如下：".format(len(FD_list)))
    for fd in FD_list:
        print(fd)
    return models


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
        # with torch.no_grad():
        outputs = FD_model(x)
        loss = criterion(outputs, y)
        # loss = mse(outputs, y_code)
        all_loss += loss
    return all_loss


def get_FD_model_Tree_GroundTruth_From_Complete_Data(ori_data, categorical_cols, zero_feed_data, fields, device, sort_corr_dict, param):
    """
    从完整数据中挖掘Ground Truth FD
    参数:
    - ori_data: 完整的原始数据 (pandas.DataFrame)
    - categorical_cols: 分类属性列的索引列表
    - zero_feed_data: 零填充的数据 (tensor)
    - fields: 字段信息
    - device: 设备
    - sort_corr_dict: 相关性排序字典
    - param: 参数字典
    """
    value_cat = []
    col_dict = {}
    data = ori_data.values
    values = ori_data.columns

    # 对于完整数据，所有行都有数据
    num_rows = data.shape[0]
    has_data_index = [list(range(num_rows)) for _ in range(len(values))]
    observer_data = [data[:, i].tolist() for i in range(len(values))]

    encoder_data = []
    new_sort_corr_dict = {}
    for col_index, col_val in enumerate(data.T):
        arr = np.array(col_val)
        cur_corr_sort = []
        if col_index in categorical_cols:
            arr = arr.astype(np.str_)
            unique_values, encoded_arr = np.unique(arr, return_inverse=True)
            encoded_arr += 1
            value_cat.append(ori_data.columns[col_index])
        else:
            # bins = np.linspace(arr.min(), arr.max(), 10)
            # encoded_arr = np.digitize(arr, bins)
            bins = np.array_split(np.sort(arr), 10)
            encoded_arr = np.zeros_like(arr)
            for i, bin_ in enumerate(bins):
                encoded_arr[np.isin(arr, bin_)] = i
        col_dict[ori_data.columns[col_index]] = col_index
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
    FD_list = []
    for key, value in col_tree.items():
        if len(value) > 0:
            y_index = col_dict[key]
            # x_index_list = get_x_index(value)
            x_index_list = value
            for x_index in x_index_list:
                new_x_index = [values[i] for i in x_index]
                FD_list.append('{}---->{}'.format(new_x_index, values[y_index]))
                model = get_model_by_tree(x_index, y_index, zero_feed_data, has_data_index, fields, device)
                if model is not None:
                    models.append(model)
    print("Ground Truth FD数量为：{}\n具体的FD如下：".format(len(FD_list)))
    for fd in FD_list:
        print(fd)

    with open('out/FD_list/{}_truth.txt'.format(param['name']), 'w') as f:
        for fd in FD_list:
            f.write(fd + '\n')
    return models

def get_FD_model_Tree(miss_data, data_m, categorical_cols, zero_feed_data, fields, device, sort_corr_dict,param):
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
    FD_list = []
    for key, value in col_tree.items():
        if len(value) > 0:
            y_index = col_dict[key]
            # x_index_list = get_x_index(value)
            x_index_list = value
            for x_index in x_index_list:
                new_x_index = [values[i] for i in x_index]
                FD_list.append('{}---->{}'.format(new_x_index, values[y_index]))
                model = get_model_by_tree(x_index, y_index, zero_feed_data, has_data_index, fields, device)
                if model is not None:
                    models.append(model)
    print("所有的FD数量为：{}\n具体的FD如下：".format(len(FD_list)))
    for fd in FD_list:
        print(fd)

    with open('out/FD_list/{}.txt'.format(param['name']), 'w') as f:
        for fd in FD_list:
            f.write(fd + '\n')
    # return None
    return models


def get_CFD_model_Tree(miss_data,data_m,categorical_cols,zero_feed_data,fields,device,sort_corr_dict,param,min_rows=5,max_pattern_size=None):
    """
    CFD version of FD mining with minimal modification.
    Discovers Conditional Functional Dependencies (Z → A | pattern)
    where pattern is generated from Z's categorical value space.

    Returns: models, dict_A, dict_B, dict_cnt
    - dict_A: key=(LHS, RHS), value=pattern (高质量CFD)
    - dict_B: key=(LHS, RHS), value=list of (pattern, support) 按support降序 (低质量CFD)
    - dict_cnt: key=(LHS, RHS), value=该结构对应的CFD数量
    """
    # =====================================================
    # Step 0. Prepare observed indices (same as FD)
    # =====================================================
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

    # =====================================================
    # Step 1. Encode data (same as FD)
    # =====================================================
    encoder_data = []
    col_dict = {}
    value_cat = []
    new_sort_corr_dict = {}

    for col_index, col_val in enumerate(data.T):
        arr = np.array(col_val)

        if col_index in categorical_cols:
            arr = arr.astype(np.str_)
            _, encoded_arr = np.unique(arr, return_inverse=True)
            encoded_arr += 1
            value_cat.append(miss_data.columns[col_index])
        else:
            bins = np.array_split(np.sort(arr), 10)
            encoded_arr = np.zeros_like(arr)
            for i, bin_ in enumerate(bins):
                encoded_arr[np.isin(arr, bin_)] = i

        col_dict[miss_data.columns[col_index]] = col_index

        # only keep categorical attrs in correlation order
        cur_corr_sort = []
        for i in sort_corr_dict[col_index]:
            if i in categorical_cols and i != col_index:
                cur_corr_sort.append(i)

        new_sort_corr_dict[col_index] = cur_corr_sort
        encoder_data.append(encoded_arr)

    encoder_data = np.array(encoder_data, dtype=np.int64)
    # =====================================================
    # Step 3. CFD mining (Tree reused, scope restricted)
    # =====================================================
    col_tree = {}
    cfd_patterns = {}

    # Progress bar for target attributes A
    for col_name in tqdm(value_cat, desc="Processing target attributes"):
        y_index = col_dict[col_name]

        Z_candidates = new_sort_corr_dict[y_index]
        if len(Z_candidates) == 0:
            continue

        featLabels_all = []

        # Build dependency tree in global space (same as FD)
        graph = {}
        for node in Z_candidates:
            graph[node] = Z_candidates[
                Z_candidates.index(node) + 1:
            ]

        # Use CTANE-style level-wise enumeration integrated with pattern enumeration:
        # buildCTANE_CFD will return a list of (z_set, valid_pattern_list) entries.
        featLabels_all = buildCTANE_CFD(
            encoder_data,
            col_dict,
            col_name,
            has_data_index,
            miss_data,
            Z_candidates,
            min_rows,
            max_pattern_size
        )

        if len(featLabels_all) > 0:
            col_tree[col_name] = featLabels_all

    # =====================================================
    # Step 4. Train models & record CFDs
    # =====================================================
    models = []
    CFD_list = []
    dict_A = {}  # key=(LHS, RHS), value=pattern (高质量CFD)
    dict_B = {}  # key=(LHS, RHS), value=list of (pattern, support) 按support降序 (低质量CFD)
    dict_cnt = {}  # key=(LHS, RHS), value=该结构对应的CFD数量

    for y_name, z_pattern_list in col_tree.items():
        y_index = col_dict[y_name]

        for z_set, pattern_list in z_pattern_list:
            for p, pat_rows in pattern_list:
                support = len(pat_rows)  # 计算support为满足pattern的行数
                # Expand pattern to include all LHS attributes (in z_set order) and RHS.
                x_names = [values[i] for i in z_set]
                entries = []
                lhs_all_wildcard = True
                for k in z_set:
                    if k in p:
                        entries.append(f"{values[k]}={p[k]}")
                        lhs_all_wildcard = False
                    else:
                        entries.append(f"{values[k]}=_")
                # RHS entry (always include, use '_' if absent)
                rhs_val = p.get(y_index, "_")
                entries.append(f"{values[y_index]}={rhs_val}")

                # Delete degenerate rules where all LHS are wildcard AND pattern is not empty
                # (we keep the truly empty pattern {} which represents all wildcards / pure FD check)
                if lhs_all_wildcard and len(p) != 0:
                    continue

                pattern_str = ",".join(entries)
                CFD_list.append(f"{x_names} ----> {values[y_index]} | {pattern_str}")

                # 构建pattern列表
                pattern_list = []
                for k in z_set:
                    if k in p:
                        pattern_list.append(str(p[k]))
                    else:
                        pattern_list.append('_')
                pattern_list.append(p.get(y_index, "_"))  # RHS的pattern

                # 根据support填充dict_A和dict_B
                structure_key = (tuple(z_set), y_index)

                if support >= min_rows:
                    # 高质量CFD，存入dict_A
                    dict_A[structure_key] = pattern_list
                else:
                    # 低质量CFD，存入dict_B
                    if structure_key not in dict_B:
                        dict_B[structure_key] = []
                    dict_B[structure_key].append((pattern_list, support))
                    # 按support降序排序
                    dict_B[structure_key].sort(key=lambda x: x[1], reverse=True)

                # 更新dict_cnt
                dict_cnt[structure_key] = dict_cnt.get(structure_key, 0) + 1

                model = get_CFD_model_by_tree(
                    z_set,
                    y_index,
                    pattern_list,
                    zero_feed_data,
                    has_data_index,
                    fields,
                    device
                )

                if model is not None:
                    models.append(model)

    # =====================================================
    # Step 5. Output
    # =====================================================
    print(f"所有的CFD数量为：{len(CFD_list)}\n具体的CFD如下：")
    for cfd in CFD_list:
        print(cfd)

    with open(f"out/CFD_list/{param['name']}.txt", "w") as f:
        for cfd in CFD_list:
            f.write(cfd + "\n")

    return models, dict_A, dict_B, dict_cnt



def get_CFD_GroundTruth_model_Tree(ori_data, categorical_cols, zero_feed_data, fields, device, sort_corr_dict, param, min_rows, max_pattern_size=None):
    """
    Discover ground-truth CFDs on a complete (no-missing) dataset `ori_data`.
    Similar algorithm to get_CFD_model_Tree but assumes all attributes are observed.

    Returns trained models for each discovered CFD and writes CFD list to disk.
    """
    data = ori_data.values
    values = ori_data.columns

    # since ori_data is complete, every column is observed in all rows
    n_rows = len(ori_data)
    has_data_index = [list(range(n_rows)) for _ in range(len(values))]

    # encode data (same style as other functions)
    encoder_data = []
    col_dict = {}
    value_cat = []
    new_sort_corr_dict = {}

    for col_index, col_val in enumerate(data.T):
        arr = np.array(col_val)
        if col_index in categorical_cols:
            arr = arr.astype(np.str_)
            _, encoded_arr = np.unique(arr, return_inverse=True)
            encoded_arr += 1
            value_cat.append(ori_data.columns[col_index])
        else:
            bins = np.array_split(np.sort(arr), 10)
            encoded_arr = np.zeros_like(arr)
            for i, bin_ in enumerate(bins):
                encoded_arr[np.isin(arr, bin_)] = i

        col_dict[ori_data.columns[col_index]] = col_index

        # restrict correlation order to categorical attributes
        cur_corr_sort = []
        for i in sort_corr_dict[col_index]:
            if i in categorical_cols and i != col_index:
                cur_corr_sort.append(i)
        new_sort_corr_dict[col_index] = cur_corr_sort
        encoder_data.append(encoded_arr)

    encoder_data = np.array(encoder_data, dtype=np.int64)

    # pattern generator (domain restricted to Z_cols)

    col_tree = {}
    # iterate over categorical target attributes (A)
    for col_name in tqdm(value_cat, desc="GT: Processing target attributes"):
        y_index = col_dict[col_name]
        Z_candidates = new_sort_corr_dict[y_index]
        if len(Z_candidates) == 0:
            continue

        # Use the same CTANE-based CFD enumeration as get_CFD_model_Tree,
        # but operate on the complete ori_data (patterns generated from ori_data).
        featLabels_all = buildCTANE_CFD(
            encoder_data,
            col_dict,
            col_name,
            has_data_index,
            ori_data,
            Z_candidates,
            min_rows,
            max_pattern_size
        )

        if len(featLabels_all) > 0:
            col_tree[col_name] = featLabels_all

    # train models and collect ground-truth CFDs (same structure as get_CFD_model_Tree)
    models = []
    CFD_list = []
    dict_A = {}
    dict_B = {}
    dict_cnt = {}

    for y_name, z_pattern_list in col_tree.items():
        y_index = col_dict[y_name]
        for z_set, pattern_list in z_pattern_list:
            for p, pat_rows in pattern_list:
                support = len(pat_rows)
                # Expand pattern to include all LHS attributes (in z_set order) and RHS.
                x_names = [values[i] for i in z_set]
                entries = []
                lhs_all_wildcard = True
                for k in z_set:
                    if k in p:
                        entries.append(f"{values[k]}={p[k]}")
                        lhs_all_wildcard = False
                    else:
                        entries.append(f"{values[k]}=_")
                rhs_val = p.get(y_index, "_")
                entries.append(f"{values[y_index]}={rhs_val}")
                # remove degenerate rules where LHS provides no constraint AND pattern is not empty
                if lhs_all_wildcard and len(p) != 0:
                    continue
                pattern_str = ",".join(entries)
                CFD_list.append(f"{x_names} ----> {values[y_index]} | {pattern_str}")

                # 构建pattern列表
                pattern_list_vals = []
                for k in z_set:
                    if k in p:
                        pattern_list_vals.append(str(p[k]))
                    else:
                        pattern_list_vals.append('_')
                pattern_list_vals.append(p.get(y_index, "_"))  # RHS的pattern

                # 根据support填充dict_A和dict_B
                structure_key = (tuple(z_set), y_index)
                if support >= min_rows:
                    dict_A[structure_key] = pattern_list_vals
                else:
                    if structure_key not in dict_B:
                        dict_B[structure_key] = []
                    dict_B[structure_key].append((pattern_list_vals, support))
                    dict_B[structure_key].sort(key=lambda x: x[1], reverse=True)

                dict_cnt[structure_key] = dict_cnt.get(structure_key, 0) + 1

                model = get_model_by_tree(z_set, y_index, zero_feed_data, has_data_index, fields, device)
                if model is not None:
                    models.append(model)

    # output (groundtruth)
    out_name = param['name'] if isinstance(param, dict) and 'name' in param else 'groundtruth'
    print(f"GT: 所有的CFD数量为：{len(CFD_list)}\n具体的CFD如下：")
    for cfd in CFD_list:
        print(cfd)
    with open(f"out/CFD_list/{out_name}_groundtruth.txt", "w") as f:
        for cfd in CFD_list:
            f.write(cfd + "\n")

    return models, dict_A, dict_B, dict_cnt

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


def get_CFD_model_by_tree(lhs, rhs, pattern, zero_feed_data, has_data_index, fields, device):
    """
    创建CFDModel对象
    """
    set_index = set(has_data_index[rhs])
    for x in lhs:
        set_index = set_index.intersection(set(has_data_index[x]))
    set_index = np.array(list(set_index))

    # 计算输入维度
    begin_list = [0]
    begin = 0
    for index, field in enumerate(fields):
        if field.data_type == "Categorical Data":
            begin += len(field.dict)
        else:
            begin += 1
        begin_list.append(begin)

    input_dim = sum(begin_list[x+1] - begin_list[x] for x in lhs)
    output_dim = begin_list[rhs+1] - begin_list[rhs]

    # 创建CFDModel
    model = CFDModel(input_dim, output_dim, lhs, rhs, pattern).to(device)

    # 训练模型（如果有训练数据）
    if len(set_index) > 0:
        X = []
        for x in lhs:
            cur_data = zero_feed_data[set_index]
            cur_data = cur_data[:, begin_list[x]:begin_list[x+1]]
            X.append(cur_data)
        X = torch.cat(X, dim=1).to(device)

        Y = zero_feed_data[set_index]
        Y = Y[:, begin_list[rhs]:begin_list[rhs+1]]
        Y = torch.argmax(Y, dim=1).long().to(device)

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
    # print(f"开始挖掘属性 '{col_name}' 的函数依赖关系...")
    for node in graph.keys():
        cur_node = []
        DFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph, cur_node, node)
    # print(f"属性 '{col_name}' 的挖掘完成，发现 {len(featLabels)} 个FD组合")

def DFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph, cur_node, start_node):
    # if col_dict[col_name] == 1:
    #     print(1)
    cur_node.append(start_node)
    if isCurNodeInFDs(cur_node, featLabels):
        return
    if len(cur_choose_row_index) == 0:
        return
    if len(cur_node) == FD_TAU:
        return
    if isCurNodeFD(encoder_data, col_dict, col_name, has_data_index, cur_node):
        new_node = cur_node.copy()
        # 获取列名映射
        reverse_col_dict = {v: k for k, v in col_dict.items()}
        left_attrs = [reverse_col_dict[node] for node in cur_node]
        # print(f"发现FD: {left_attrs} --> {col_name}")
        updateFeatLabels(new_node, featLabels)
        return
    for node in graph[start_node]:
        if node not in cur_node:
            start_node = node
            DFS(encoder_data, col_dict, col_name, has_data_index, featLabels, cur_choose_row_index, graph,
                         cur_node, start_node)

            cur_node.pop()


def generate_patterns_for_Z_global(miss_data, Z_cols, min_rows, max_k=None, rhs_index=None):
    """
    Module-level pattern generator usable by buildCTANE_CFD.
    Returns list of (pattern_dict, rows_set).
    """
    total = len(miss_data)
    # Precompute, for each attribute in Z_cols, value -> set(rows)
    val_rows = {}
    for z in Z_cols:
        col_vals, counts = np.unique(miss_data.iloc[:, z], return_counts=True)
        for v, c in zip(col_vals, counts):
            if c >= min_rows:
                rows = set(miss_data.index[miss_data.iloc[:, z] == v])
                val_rows.setdefault(z, []).append((v, rows))

    if max_k is None:
        max_k = len(Z_cols)
    else:
        max_k = min(max_k, len(Z_cols))

    # Level 1 patterns
    patterns_level = []
    # include RHS column values if requested (domain = Z_cols U {rhs_index})
    full_cols = list(Z_cols)
    if rhs_index is not None and rhs_index not in full_cols:
        full_cols.append(rhs_index)

    for z, entries in val_rows.items():
        for v, rows in entries:
            patterns_level.append(({z: v}, rows))

    all_patterns = patterns_level.copy()

    # Iteratively build larger patterns up to max_k
    for k in range(2, min(max_k, len(full_cols)) + 1):
        next_level = []
        for pat_dict, pat_rows in patterns_level:
            pat_attrs = set(pat_dict.keys())
            for z in full_cols:
                if z in pat_attrs:
                    continue
                entries = val_rows.get(z, [])
                for v, rows_z in entries:
                    new_rows = pat_rows & rows_z
                    if len(new_rows) < min_rows:
                        continue
                    new_pat = pat_dict.copy()
                    new_pat[z] = v
                    next_level.append((new_pat, new_rows))
        if len(next_level) == 0:
            break
        all_patterns.extend(next_level)
        patterns_level = next_level

    return all_patterns


def generate_patterns_for_Z_encoded(encoder_data, miss_data, col_dict, Z_cols, min_rows, max_k=None):
    """
    Generate patterns based on encoder_data (numeric codes) but return patterns
    with readable raw values and row sets.
    - encoder_data: numpy array shape (n_cols, n_rows) of encoded values (ints)
    - miss_data: original DataFrame used to retrieve raw display values
    - col_dict: mapping column name -> index
    - Z_cols: list of column indices to consider
    """
    total = len(miss_data)
    # map encoded value -> rows for each z
    val_rows = {}
    # also map encoded value -> raw display string (prefer categorical original)
    val_display = {}
    for z in Z_cols:
        col_vals = encoder_data[z]
        unique_vals, counts = np.unique(col_vals, return_counts=True)
        for ev, c in zip(unique_vals, counts):
            if c >= min_rows:
                rows = set(np.where(col_vals == ev)[0])
                val_rows.setdefault(z, []).append((ev, rows))
                # find a representative raw value from miss_data (if available)
                raw_candidates = np.unique(miss_data.iloc[:, z].values[col_vals == ev])
                if len(raw_candidates) > 0:
                    # pick first non-null-like representation
                    rep = None
                    for rc in raw_candidates:
                        if rc is None:
                            continue
                        s = str(rc).strip()
                        if s == '' or s.lower() in ('null', 'none'):
                            continue
                        rep = s
                        break
                    if rep is None:
                        rep = str(raw_candidates[0])
                else:
                    rep = f"code_{ev}"
                val_display.setdefault(z, {})[ev] = rep

    # include RHS column if provided via col_dict mapping in caller by adding it to Z_cols before calling
    if max_k is None:
        max_k = len(Z_cols)
    else:
        max_k = min(max_k, len(Z_cols))

    # Level 1 patterns (use display values as pattern values)
    patterns_level = []
    for z, entries in val_rows.items():
        for ev, rows in entries:
            patt = {z: val_display[z][ev]}
            patterns_level.append((patt, rows))

    all_patterns = patterns_level.copy()

    for k in range(2, min(max_k, len(Z_cols)) + 1):
        next_level = []
        for pat_dict, pat_rows in patterns_level:
            pat_attrs = set(pat_dict.keys())
            for z, entries in val_rows.items():
                if z in pat_attrs:
                    continue
                for ev, rows_z in entries:
                    new_rows = pat_rows & rows_z
                    if len(new_rows) < min_rows:
                        continue
                    new_pat = pat_dict.copy()
                    new_pat[z] = val_display[z][ev]
                    next_level.append((new_pat, new_rows))
        if len(next_level) == 0:
            break
        all_patterns.extend(next_level)
        patterns_level = next_level

    return all_patterns

def buildCTANE_CFD(encoder_data, col_dict, col_name, has_data_index, miss_data, Z_candidates, min_rows, max_pattern_size):
    """
    Level-wise enumeration over Z_candidates that directly enumerates patterns (tp)
    for each candidate Z and validates CFDs (Z -> A | tp) on the pattern's row subset.

    Returns: list of tuples (z_set, valid_patterns) where valid_patterns is list of (pattern_dict, z_set).
    """
    featLabels_results = []

    # start with singletons
    candidates = [[z] for z in Z_candidates]
    level = 1
    max_level = FD_TAU

    def is_covered_by_existing(z_set):
        for exist_z, _ in featLabels_results:
            if set(exist_z) <= set(z_set):
                return True
        return False

    while candidates and level <= max_level:
        next_level = []
        for cand in candidates:
            # NOTE: minimality pruning disabled temporarily per request.
            # Previously we skipped candidates covered by an existing smaller Z:
            if is_covered_by_existing(cand):
                continue

            # prepare pattern domain: cand U {y_index}
            y_index = col_dict[col_name]
            cand_with_rhs = cand.copy()
            if y_index not in cand_with_rhs:
                cand_with_rhs = cand_with_rhs + [y_index]

            # determine max pattern size for this cand,default very large
            mpk = max_pattern_size if max_pattern_size is not None else 1000

            # Level-wise pattern enumeration and validation:
            #  - First check all 1-item patterns (k=1). If a 1-item pattern already
            #    validates as CFD on the rows, record it and DO NOT expand it.
            #  - Only patterns that do NOT validate at level k are joined among themselves
            #    to form candidates for level k+1.

            # First, consider the empty pattern (all wildcards) as a valid candidate.
            # empty pattern rows = all rows (will be intersected with base_rows later)
            patterns_level = [({}, set(range(len(miss_data))))]
            # then add level-1 patterns (using existing generator with max_k=1)
            patterns_level.extend(generate_patterns_for_Z_encoded(encoder_data, miss_data, col_dict, cand_with_rhs, min_rows, max_k=1))
            # patterns_level: list of (pat_dict, rows)

            valid_patterns = []
            non_established = []
            base_rows = set(has_data_index[y_index])

            # validate level-1 patterns first
            for p_dict, p_rows in patterns_level:
                rows_p = base_rows & p_rows
                if len(rows_p) < min_rows:
                    continue
                choose_row_index = np.array(list(rows_p))
                if isCurNodeFD_InRows(encoder_data, col_dict, col_name, has_data_index, cand, choose_row_index):
                    valid_patterns.append((p_dict, cand))
                else:
                    non_established.append((p_dict, p_rows))

            # If the empty pattern (all wildcards) validates, treat it as the global FD
            # for this Z and do not search for or record any more specific patterns.
            try:
                empty_valid = any(p_dict == {} for p_dict, _ in valid_patterns)
            except Exception:
                empty_valid = False
            if empty_valid:
                # keep only the empty pattern entry
                valid_patterns = [vp for vp in valid_patterns if vp[0] == {}]
                featLabels_results.append((cand, valid_patterns))
                # do not expand this candidate further
                continue

            # expand to higher-order patterns only from non_established patterns
            k = 1
            while k < mpk and len(non_established) > 0:
                next_non_established = []
                seen_keys = set()
                n = len(non_established)
                for i in range(n):
                    for j in range(i + 1, n):
                        pat_i, rows_i = non_established[i]
                        pat_j, rows_j = non_established[j]
                        keys_i = set(pat_i.keys())
                        keys_j = set(pat_j.keys())
                        # join only if their union increases by one attribute and they don't conflict
                        if len(keys_i | keys_j) != k + 1:
                            continue
                        conflict = False
                        for kk in keys_i & keys_j:
                            if pat_i[kk] != pat_j[kk]:
                                conflict = True
                                break
                        if conflict:
                            continue
                        union_keys = tuple(sorted(list(keys_i | keys_j)))
                        if union_keys in seen_keys:
                            continue
                        seen_keys.add(union_keys)
                        # merge pattern dicts
                        new_pat = pat_i.copy()
                        new_pat.update(pat_j)
                        new_rows = rows_i & rows_j
                        if len(new_rows) < min_rows:
                            continue
                        # validate this new pattern
                        rows_p = base_rows & new_rows
                        if len(rows_p) < min_rows:
                            continue
                        choose_row_index = np.array(list(rows_p))
                        if isCurNodeFD_InRows(encoder_data, col_dict, col_name, has_data_index, cand, choose_row_index):
                            valid_patterns.append((new_pat, cand))
                        else:
                            next_non_established.append((new_pat, new_rows))
                non_established = next_non_established
                k += 1

            if valid_patterns:
                featLabels_results.append((cand, valid_patterns))
            next_level.append(cand)

        # join step to create next-level Z candidates (same as before)
        new_candidates = []
        n = len(next_level)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                a = next_level[i]
                b = next_level[j]
                union = sorted(set(a) | set(b))
                if len(union) == level + 1:
                    key = tuple(union)
                    if key not in seen:
                        seen.add(key)
                        new_candidates.append(union)

        candidates = new_candidates
        level += 1

    return featLabels_results


def updateFeatLabels(new_node, featLabels):
    #判断featLabels中是否有它的子集
    node_list = []
    for node in featLabels:
        if set(new_node) <= set(node) and len(new_node) > 0:
            node_list.append(node)
    if len(node_list) > 0:
        # print(f"  移除冗余FD: {[n for n in node_list]} (被新FD {new_node} 包含)")
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
    # 判断左边划分的数据在右侧划分中的数据是否是一致的
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


def isCurNodeFD_InRows(encoder_data, col_dict, col_name, has_data_index, cur_node, row_subset):
    """
    Check if FD holds in a specific subset of rows (used for CFD validation).
    Similar to isCurNodeFD but restricted to row_subset.
    """
    col_index = col_dict[col_name]
    right_observe_index = has_data_index[col_index]
    # Intersect with row_subset
    observe_index = list(set(right_observe_index).intersection(set(row_subset)))

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

    # 判断左边划分的数据在右侧划分中的数据是否是一致的
    for value, probability in zip(unique_values, probabilities):
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
    if len(x_index_list) > FD_TAU:
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
        #if len(FD.x_index_list) == 1 and FD.x_index_list[0] == 0 and FD.y_index == 2:
        #   print('FD score：{}, against FD tuples：{}, against FD tuples score：{}，meet FD tuples：{}，meet FD tuples score：{}'.format(FD_score,numpy_impute_data[un_satisfy_tup][all_index], un_sat_score,numpy_impute_data[all_satisfy_tup_inS][all_index], all_satisfy_tup_inS_acc))
        old_FD_score = FD_score
        if FD_score-un_sat_score < 0:
            filename = 'result.txt'
            print('remove FD model_label_index:{},    model_feature_index:{}'.format(FD.y_index, FD.x_index_list))
            # print('FD score：{}, against FD tuples：{}, against FD tuples score：{}，meet FD tuples：{}，meet FD tuples score：{}'.format(FD_score,numpy_impute_data[un_satisfy_tup][all_index], un_sat_score,numpy_impute_data[all_satisfy_tup_inS][all_index], all_satisfy_tup_inS_acc))
            with open(filename, 'w') as file:
                print('remove FD model_label_index:{},    model_feature_index:{}'.format(FD.y_index, FD.x_index_list), file = file)
                # print('FD score：{}, against FD tuples：{},  against FD tuples score：{}，meet FD tuples：{}，meet FD tuples score：{}'.format(FD_score,numpy_impute_data[un_satisfy_tup], un_sat_score,numpy_impute_data[all_satisfy_tup_inS], all_satisfy_tup_inS_acc),file=file)

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


# CFD相关类和函数实现
class CFDModel(nn.Module):
    def __init__(self, input_size, output_size, lhs, rhs, pattern):
        super(CFDModel, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.lhs = lhs  # LHS属性索引列表
        self.rhs = rhs  # RHS属性索引
        self.pattern = pattern  # pattern列表，包含常量和'_'

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def set_pattern(self, new_pattern):
        self.pattern = new_pattern



def calculate_cfd_score(lhs, rhs, pattern, impute_data, eq_dict, tuple_acc_list):
    """
    计算CFD得分：在pattern子集上复用FD得分计算逻辑
    """
    # 获取满足pattern的元组，作为RES_X
    RES_X = get_pattern_satisfying_tuples(lhs + [rhs], pattern, impute_data, eq_dict)

    if len(RES_X) == 0:
        return 0.0, 0.0, '', [], []

    # 在pattern子集上复用FD的satisfy/unsatisfy分类逻辑
    satisfy_tuples, un_satisfy_tuples = get_satisfy_unsatisfy_tuples(RES_X, eq_dict, rhs, impute_data, tuple_acc_list)

    # 直接使用FD的得分计算函数
    CFD_score, max_un_sat_score, un_satisfy_tup, all_satisfy_tup_inS, all_satisfy_tup_inS_acc = get_FD_score(satisfy_tuples, un_satisfy_tuples, tuple_acc_list)

    return CFD_score, max_un_sat_score, un_satisfy_tup, all_satisfy_tup_inS, all_satisfy_tup_inS_acc


def update_CFD_models(generate_x, zero_feed_data, fields, data_m, M_tensor, value_cat, values, miss_data_x, enc, device, eq_dict, CFD_model_list, cell_acc, continuous_cols, cost_time, dict_A, dict_B, dict_cnt, min_rows, max_pattern_per_structure):
    """
    CFD refine操作：对当前CFD列表进行细化
    注意：top_k会根据dict_B的变化动态调整
    """
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

    new_CFD_model_list = CFD_model_list.copy()

    for CFD in CFD_model_list:
        lhs = CFD.lhs
        rhs = CFD.rhs
        pattern = CFD.pattern

        # 计算该CFD的属性索引列表（LHS + RHS）
        index_list = lhs.copy()
        index_list.append(rhs)

        # 计算tuple_acc_list：使用pattern-aware的cell_acc
        tuple_acc_list = []
        for tuple_index, tuple_acc in enumerate(cell_acc):
            tuple_acc_list.append(min(tuple_acc[index_list]))

        # 步骤1：在当前pattern域内计算CFD得分
        CFD_score, max_un_sat_score, un_satisfy_tup, all_satisfy_tup_inS, all_satisfy_tup_inS_acc = calculate_cfd_score(lhs, rhs, pattern, impute_data, new_eq_dict, tuple_acc_list)

        # 步骤2：若CFD_score - max_un_sat_score < 0，进行refine（类似FD的判断逻辑）
        if CFD_score - max_un_sat_score < 0:
            structure_key = (tuple(lhs), rhs)

            # 步骤2.1：查询dict_cnt[(LHS, RHS)]
            if dict_cnt.get(structure_key, 0) > max_pattern_per_structure:
                # 从dict_A和CFD列表中删除当前CFD（终态）
                if structure_key in dict_A:
                    del dict_A[structure_key]
                new_CFD_model_list.remove(CFD)

                # 从dict_B[(LHS, RHS)]中按support降序枚举top-K个pattern
                if structure_key in dict_B:
                    success = False

                    # 计算本结构下满足 support >= min_rows/2 的候选pattern数量，作为当前 top_k
                    half_threshold = min_rows / 2.0
                    eligible_count = sum(1 for (_p, s) in dict_B[structure_key] if s >= half_threshold)
                    current_top_k = max(1, eligible_count)

                    candidate_patterns = dict_B[structure_key][:current_top_k]
                    for i, (cand_pattern, support) in enumerate(candidate_patterns):
                        # 计算候选pattern的CFD得分
                        cand_CFD_score, cand_max_un_sat_score, _, _, _ = calculate_cfd_score(lhs, rhs, cand_pattern, impute_data, new_eq_dict, tuple_acc_list)

                        if support >= min_rows and cand_CFD_score > CFD_score:
                            # 将该pattern从dict_B中删除
                            dict_B[structure_key].remove((cand_pattern, support))
                            # 将其加入dict_A
                            dict_A[structure_key] = cand_pattern
                            success = True
                            break

                    # 若top-K均失败，则直接结束该结构的refine
                    if not success:
                        continue
                else:
                    continue

            # 步骤2.2：若结构数量未超过阈值，对当前pattern执行specialization
            else:
                specialized = False

                # 尝试pattern specialization：将'_'替换为具体属性值
                specialized_pattern, spec_CFD_score = pattern_specialization(lhs, rhs, pattern, impute_data, new_eq_dict, cell_acc, tuple_acc_list, min_rows)

                if specialized_pattern is not None and spec_CFD_score > CFD_score:
                    CFD.set_pattern(specialized_pattern)
                    specialized = True

                # 步骤2.3：若pattern specialization失败，执行LHS augmentation
                if not specialized:
                    augmented = False

                    # 逐步向LHS中加入新属性，新属性对应的pattern从'_'开始枚举
                    augmented_lhs, augmented_pattern, aug_CFD_score = lhs_augmentation(lhs, rhs, pattern, impute_data, new_eq_dict, cell_acc, tuple_acc_list, values, continuous_cols, min_rows)

                    if augmented_lhs is not None and aug_CFD_score > CFD_score:
                        CFD.lhs = augmented_lhs
                        CFD.set_pattern(augmented_pattern)
                        augmented = True

                    # 若所有LHS扩展均失败，则删除该CFD（终态）
                    if not augmented:
                        structure_key = (tuple(lhs), rhs)
                        if structure_key in dict_A:
                            del dict_A[structure_key]
                        new_CFD_model_list.remove(CFD)

        # 步骤3：若c(φ) ≥ c(φ*)，认为该CFD成立，更新dict_A中对应的pattern
        else:
            structure_key = (tuple(lhs), rhs)
            dict_A[structure_key] = pattern

    return new_CFD_model_list


def calculate_cfd_confidence(lhs, rhs, pattern, impute_data, eq_dict, cell_acc):
    """
    计算CFD的置信度，只在pattern条件对应的子表上计算
    """
    # 获取满足pattern的元组
    pattern_tuples = get_pattern_satisfying_tuples(lhs + [rhs], pattern, impute_data, eq_dict)

    if len(pattern_tuples) == 0:
        return 0.0

    # 计算满足FD的元组数量（LHS相同且RHS相同），使用cell_acc进行加权
    satisfy_score = 0.0
    total_score = 0.0

    for tuples in pattern_tuples:
        if len(tuples) > 1:
            # 检查LHS是否相同且RHS是否相同
            lhs_values = [impute_data.iloc[t, lhs[0]] for t in tuples] if len(lhs) == 1 else []
            if len(lhs) > 1:
                lhs_values = [[impute_data.iloc[t, attr] for attr in lhs] for t in tuples]
            rhs_values = [impute_data.iloc[t, rhs] for t in tuples]

            # 检查是否所有LHS相同且所有RHS相同
            lhs_same = all(val == lhs_values[0] for val in lhs_values) if lhs_values else True
            rhs_same = all(val == rhs_values[0] for val in rhs_values)

            # 使用cell_acc对每个元组进行加权评分
            tuple_score = sum(cell_acc[tuple_idx, attr] for tuple_idx in tuples for attr in lhs + [rhs]) / len(tuples) / len(lhs + [rhs])

            if lhs_same and rhs_same:
                satisfy_score += tuple_score
            total_score += tuple_score
        else:
            # 单个元组的情况
            tuple_score = sum(cell_acc[tuple_idx, attr] for tuple_idx in tuples for attr in lhs + [rhs]) / len(tuples) / len(lhs + [rhs])
            total_score += tuple_score

    confidence = satisfy_score / total_score if total_score > 0 else 0.0
    return confidence


def get_pattern_satisfying_tuples(attrs, pattern, impute_data, eq_dict):
    """
    获取满足pattern的元组分组
    """
    pattern_tuples = []

    # 从最小的等价类开始分组
    min_attr = min(attrs)
    if min_attr in eq_dict:
        for val, tuples in eq_dict[min_attr].items():
            if len(tuples) > 1:
                # 检查pattern匹配
                if check_pattern_match(tuples, attrs, pattern, impute_data):
                    pattern_tuples.append(tuples)

    return pattern_tuples


def check_pattern_match(tuples, attrs, pattern, impute_data):
    """
    检查元组组是否满足pattern
    """
    if len(tuples) == 0:
        return False

    for i, attr in enumerate(attrs):
        if pattern[i] != '_':
            # 检查所有元组在该属性上的值是否都等于pattern[i]
            attr_values = [impute_data.iloc[t, attr] for t in tuples]
            if not all(val == pattern[i] for val in attr_values):
                return False

    return True


def recalculate_cfd_metrics(lhs, rhs, pattern, impute_data, eq_dict, cell_acc):
    """
    重新计算CFD的support和confidence
    """
    pattern_tuples = get_pattern_satisfying_tuples(lhs + [rhs], pattern, impute_data, eq_dict)

    # 使用cell_acc计算加权support
    support = 0.0
    for tuples in pattern_tuples:
        # 对每个元组组，使用cell_acc进行加权
        tuple_score = sum(cell_acc[tuple_idx, attr] for tuple_idx in tuples for attr in lhs + [rhs]) / len(tuples) / len(lhs + [rhs])
        support += tuple_score

    confidence = calculate_cfd_confidence(lhs, rhs, pattern, impute_data, eq_dict, cell_acc)

    return support, confidence


def pattern_specialization(lhs, rhs, pattern, impute_data, eq_dict, cell_acc, tuple_acc_list, min_rows):
    """
    pattern specialization：将'_'替换为具体属性值
    """
    best_pattern = None
    best_CFD_score = -1.0

    for i, val in enumerate(pattern):
        if val == '_':
            attr = lhs[i] if i < len(lhs) else rhs
            # 尝试将'_'替换为该属性上的具体值
            unique_vals = impute_data.iloc[:, attr].unique()

            for concrete_val in unique_vals:
                new_pattern = pattern.copy()
                new_pattern[i] = concrete_val

                # 计算新pattern的support和CFD得分
                support, _ = recalculate_cfd_metrics(lhs, rhs, new_pattern, impute_data, eq_dict, cell_acc)
                new_CFD_score, _, _, _, _ = calculate_cfd_score(lhs, rhs, new_pattern, impute_data, eq_dict, tuple_acc_list)

                if support >= min_rows and new_CFD_score > best_CFD_score:
                    best_CFD_score = new_CFD_score
                    best_pattern = new_pattern

    return best_pattern, best_CFD_score


def lhs_augmentation(lhs, rhs, pattern, impute_data, eq_dict, cell_acc, tuple_acc_list, values, continuous_cols, min_rows):
    """
    LHS augmentation：逐步向LHS中加入新属性
    """
    best_lhs = None
    best_pattern = None
    best_CFD_score = -1.0

    # 尝试添加新属性到LHS
    for attr_idx in range(len(values)):
        if attr_idx not in lhs and attr_idx != rhs and attr_idx not in continuous_cols:
            new_lhs = lhs + [attr_idx]
            new_pattern = pattern + ['_']  # 新属性对应的pattern从'_'开始

            # 计算新LHS的CFD得分
            new_CFD_score, _, _, _, _ = calculate_cfd_score(new_lhs, rhs, new_pattern, impute_data, eq_dict, tuple_acc_list)

            if new_CFD_score > best_CFD_score:
                best_CFD_score = new_CFD_score
                best_lhs = new_lhs
                best_pattern = new_pattern

    return best_lhs, best_pattern, best_CFD_score
