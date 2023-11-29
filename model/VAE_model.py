import pandas as pd
import torch.nn as nn
import torch.nn.functional as fun
import math
from model.FD_model import get_my_FD_loss, train_new_FD_model, get_eq_dict, update_FD_models, get_trueProObserve, true_Pro
import matplotlib.pyplot as plt
import torch
from model.Discriminator_model import D, train_discriminator
import torch.optim as optim
import numpy as np
from utils.util import get_M_by_data_m, test_impute_data_rmse, test_impute_data_acc, get_valid_data_index, sample_x, \
    categorical_to_code, Data_convert, reconvert_data, get_down_acc
from model.Learner import train_L_code
from tqdm import tqdm


def xavier(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev)





class Vae(nn.Module):
    def __init__(self, input_dim, encoder_dim, encoder_out_dim, latent_dim, fields):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=False),
            nn.Linear(encoder_dim, encoder_out_dim),
            nn.BatchNorm1d(encoder_out_dim),
            nn.ReLU(inplace=False),
        )
        self.en_to_la = nn.Linear(encoder_out_dim, latent_dim)
        self.mu = nn.Linear(encoder_out_dim, latent_dim)  # mean
        self.sigma = nn.Linear(encoder_out_dim, latent_dim)  # sigma,z为浅层的维度
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fields = fields
        self.linear_1 = nn.Linear(latent_dim, encoder_out_dim)
        self.linear_2 = nn.Linear(encoder_out_dim, encoder_dim)
        self.linear_3 = nn.Linear(encoder_dim, input_dim)
        self.batch_normal1 = nn.BatchNorm1d(encoder_out_dim)
        self.batch_normal2 = nn.BatchNorm1d(encoder_dim)
        self.relu = nn.ReLU(inplace=False)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def attention_score(self, z):
        Dimr = z.shape[1]
        device = z.device
        G_WQ = torch.tensor(xavier([Dimr, Dimr]), dtype=torch.float).to(device)
        G_b1 = torch.tensor(np.zeros(shape=[Dimr]), dtype=torch.float).to(device)

        G_WK = torch.tensor(xavier([Dimr, Dimr]), dtype=torch.float).to(device)
        G_b2 = torch.tensor(np.zeros(shape=[Dimr]), dtype=torch.float).to(device)

        G_WV = torch.tensor(xavier([Dimr, Dimr]), dtype=torch.float).to(device)
        G_b3 = torch.tensor(np.zeros(shape=[Dimr]), dtype=torch.float).to(device)
        Dimr = torch.tensor(Dimr, dtype=torch.float).to(device)
        inputs = z
        Q = fun.relu(torch.matmul(inputs, G_WQ) + G_b1)
        K = fun.relu(torch.matmul(inputs, G_WK) + G_b2)
        V = fun.relu(torch.matmul(inputs, G_WV) + G_b3)
        attn1 = fun.relu(torch.mm(V, torch.softmax(torch.matmul(Q.T, K) / torch.sqrt(Dimr), dim=1)))
        return attn1
    def parameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(std.device)
        z = mu + eps * std
        return z

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()


    def forward(self, x):
        out1 = self.encoder(x)  # out：encoder_out_dim
        z = self.en_to_la(out1)
        mu = self.mu(out1)
        log_var = self.sigma(out1)
        z_ = self.parameterize(mu, log_var)  # out: latent_dim
        # 对z_进行attention
        z_ = self.attention_score(z_)
        # decoder
        out = self.linear_1(z_)
        if x.dim() == 2:
            out = self.batch_normal1(out)
        else:
            m, n, k = out.size()
            matrix = out.transpose(1, 2).contiguous().view(-1, k)
            output_matrix = self.batch_normal1(matrix)
            out = output_matrix.view(m, n, k)
        out = self.relu(out)
        out = self.linear_2(out)
        if x.dim() == 2:
            out = self.batch_normal2(out)
        else:
            m, n, k = out.size()
            matrix = out.transpose(1, 2).contiguous().view(-1, k)
            output_matrix = self.batch_normal2(matrix)
            out = output_matrix.view(m, n, k)
        out = self.relu(out)
        out = self.linear_3(out)  # out: input_dim
        # 对类别数据做softmax
        current_ind = 0
        decodes = []
        out_decodes = []
        if x.dim() == 2:
            for i in range(len(self.fields)):
                if self.fields[i].data_type == "Categorical Data":
                    dim = self.fields[i].dim()  # one-hot向量维度
                    data = fun.softmax(out[:, current_ind:current_ind + dim], dim=1)
                    decodes.append(data)
                    max_index = torch.argmax(data, dim=1)
                    one_hot = torch.zeros_like(data)
                    one_hot.scatter_(1, max_index.unsqueeze(1), 1)
                    out_decodes.append(one_hot)
                    current_ind = current_ind + dim
                else:
                    decodes.append(self.sig(out[:, current_ind:current_ind + 1]))
                    out_decodes.append(self.sig(out[:, current_ind:current_ind + 1]))
                    current_ind = current_ind + 1
            decodes = torch.cat(decodes, dim=1)
            out_decodes = torch.cat(out_decodes, dim=1)
        else:
            for i in range(len(self.fields)):
                if self.fields[i].data_type == "Categorical Data":
                    dim = self.fields[i].dim()  # one-hot向量维度
                    data = fun.softmax(out[i, :, current_ind:current_ind + dim], dim=1)
                    decodes.append(data)
                    max_index = torch.argmax(data, dim=1)
                    one_hot = torch.zeros_like(data)
                    one_hot[:, max_index] = 1
                    out_decodes.append(one_hot)
                    current_ind = current_ind + dim
                else:
                    decodes.append(self.sig(out[:, current_ind:current_ind + 1]))
                    out_decodes.append(self.sig(out[:, current_ind:current_ind + 1]))
                    current_ind = current_ind + 1
            decodes = torch.cat(decodes, dim=1)
            out_decodes = torch.cat(out_decodes, dim=1)

        return z_, mu, log_var, decodes, out_decodes


def loss_func(fields, reconstruct, x, mu, log_var, M):
    batch_size = x.size(0)
    MSE = 0
    # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    # threshold = torch.ones_like(KLD) * 0.01
    # # 比较KL散度和阈值
    # mask = KLD > threshold
    # # 将KL散度小于阈值的部分置为0
    # KLD = KLD * mask.float()
    # KLD = KLD.mean()
    BCE = 0
    curr = 0
    MSE_num = 0
    BCE_num = 0
    for i in range(len(fields)):
        dim = fields[i].dim()
        if fields[i].data_type == 'Numerical Data':
            m = M[:, curr:curr + dim].cpu().detach().numpy()
            good_rows = np.where(np.all(m == 1, axis=1))[0]
            pre = reconstruct[good_rows, curr:curr + dim]
            label = x[good_rows, curr:curr + dim]
            # mse = torch.sum(torch.pow(reconstruct[:, curr:curr + dim] * M[:, curr:curr + dim] - x[:, curr:curr + dim] * M[:, curr:curr + dim], 2))
            # mse2 = fun.mse_loss(reconstruct[:, curr:curr + dim], x[:, curr:curr + dim], reduction="mean")
            loss_fn = nn.MSELoss()
            mse = loss_fn(pre, label)
            MSE += mse
            MSE_num = MSE_num + 1
        else:
            m = M[:, curr:curr + dim].cpu().detach().numpy()
            good_rows = np.where(np.all(m == 1, axis=1))[0]
            # bce = fun.binary_cross_entropy(reconstruct[:, curr:curr + dim], x[:, curr:curr + dim], reduction="mean")
            # bce = -torch.sum(x[:, curr:curr + dim] * torch.log(reconstruct[:, curr:curr + dim] + 1e-9) * M[:, curr:curr + dim] + (1 - x[:, curr:curr + dim]) * torch.log(1 - reconstruct[:, curr:curr + dim] + 1e-9) * M[:, curr:curr + dim])/count
            pre = reconstruct[good_rows,  curr:curr + dim]
            lab = x[good_rows, curr:curr + dim]
            label = torch.argmax(lab, dim=1)
            loss_fn = nn.MSELoss()
            loss = loss_fn(pre, lab)
            BCE += loss
            BCE_num = BCE_num + 1
        curr += dim
    if BCE_num == 0:
        BCE_num = BCE_num + 1
    if MSE_num == 0:
        MSE_num = MSE_num + 1
    # if math.isnan(KLD + (MSE / MSE_num + BCE / BCE_num) / 2):
    #     print(1)
    # 对MSE和BCE进行权衡
    return MSE/MSE_num , BCE/BCE_num, KLD


def alpha_schedule(epoch, max_epoch, alpha_max, strategy="exp"):
    # strategy to adjust weight
    if strategy == "linear":
        alpha = alpha_max * min(1, epoch / max_epoch)
    elif strategy == "exp":
        alpha = alpha_max * math.exp(-5 * (1 - min(1, epoch / max_epoch)) ** 2)
    else:
        raise NotImplementedError("Strategy {} not implemented".format(strategy))
    return alpha


def train_vae(use_discrinminator, epoch, loss_weight, optimizer, x, zero_feed_code, M_tensor, m_data, fields,L_loss, vae, discriminator):
    kl_weight = alpha_schedule(epoch, 800, 0.5)
    optimizer.zero_grad()
    z_, mu, log_var, generate_x, generate_out_x = vae(x)
    decoder_z_impute = zero_feed_code + (1 - M_tensor) * generate_x
    MSE_loss, BCE_loss, KL_loss = loss_func(fields, generate_x, x, mu, log_var,M_tensor)
    if BCE_loss != 0:
        BCE_loss = BCE_loss * loss_weight['BCE_weight']
    Reconstruction_loss = MSE_loss + BCE_loss
    # FD_loss = get_my_FD_loss(FDs_model_list, decoder_z_impute, fields)
    if use_discrinminator:
        discriminator_z = discriminator(decoder_z_impute)
        Generator_Discriminator_z_loss = (-torch.mean((1 - m_data) * torch.log(discriminator_z + 1e-8)))
    else:
        Generator_Discriminator_z_loss = 0
    generator_loss = loss_weight['VAE_Res_weight'] * Reconstruction_loss + kl_weight * KL_loss + Generator_Discriminator_z_loss * \
                         loss_weight['VAE_G_weight'] + L_loss * loss_weight['L_weight']
    # if L_loss == 0:
    #     if FD_loss == 0:
    #         generator_loss = loss_weight[
    #                              'VAE_Res_weight'] * Reconstruction_loss + kl_weight * KL_loss + Generator_Discriminator_z_loss * \
    #                          loss_weight['VAE_G_weight']
    #     else:
    #         generator_loss = loss_weight[
    #                              'VAE_Res_weight'] * Reconstruction_loss + kl_weight * KL_loss + Generator_Discriminator_z_loss * \
    #                          loss_weight['VAE_G_weight'] + FD_loss * loss_weight['FD_weight']
    # else:
    #     if FD_loss == 0:
    #         generator_loss = loss_weight['VAE_Res_weight'] * Reconstruction_loss + kl_weight * KL_loss + Generator_Discriminator_z_loss * \
    #                          loss_weight['VAE_G_weight'] + L_loss * loss_weight['L_weight']
    #     else:
    #         generator_loss = loss_weight[
    #                              'VAE_Res_weight'] * Reconstruction_loss + kl_weight * KL_loss + Generator_Discriminator_z_loss * \
    #                          loss_weight['VAE_G_weight'] + FD_loss * loss_weight['FD_weight'] + L_loss * loss_weight[
    #                              'L_weight']
    generator_loss.backward()
    optimizer.step()
    if use_discrinminator:
        return generator_loss.item(),Reconstruction_loss.item(),(kl_weight * KL_loss).item(),Generator_Discriminator_z_loss.item(),L_loss
    else:
        return generator_loss.item(),Reconstruction_loss.item(),(kl_weight * KL_loss).item(),Generator_Discriminator_z_loss,L_loss


# 输入n * dim的tensor，计算出每个值的置信度,数值类型不用计算默认为1
def get_truePro_acc(truePro_vae, truePro_observe, continuse_cols, data_m):
    # 1. 计算预测正的总和
    true_pro = np.sum(truePro_vae * truePro_observe * data_m)
    # 2. 计算预测错误的综合
    con_cols = truePro_vae[:, continuse_cols]
    falsePro_vae = 1 - truePro_vae
    falsePro_vae[:, continuse_cols] = con_cols

    con_cols = truePro_observe[:, continuse_cols]
    falsePro_observe = 1 - truePro_observe
    falsePro_observe[:, continuse_cols] = con_cols
    false_pro = np.sum(falsePro_vae * falsePro_observe * data_m)


    #3. 计算所有的正确率，并求均值返回
    sum_true = true_pro + false_pro
    new_data_m = data_m.copy()
    new_data_m[:, continuse_cols] = 0
    zhiXinDu_vae = sum_true / (np.sum(new_data_m)+1)
    return zhiXinDu_vae


def get_cell_true(generate_x, zero_feed_data, fields, data_m, continuous_cols, discriminator, generate_out_x, M_tensor, value_cat, values, miss_data_x, enc, ori_data, device):
    # 1. 获取vae的准确率
    # 1.1 根据生成的数据获得每个值的的置信度
    truePro_vae = get_trueProVAE(generate_x, fields)
    # 1.2 根据已知的数据区域获取真实的置信度
    truePro_observe = get_trueProObserve(generate_x, zero_feed_data, fields, data_m, device)
    # 1.3 判断真实的置信度与生成的置信度的差异获取准确率
    truePro_acc_vae = get_truePro_acc(truePro_vae, truePro_observe, continuous_cols, data_m)
    # 2. 获取Dis的准确率
    # 2.1 根据填充后的数据获取每个值的置信度
    truePro_dis = discriminator(generate_out_x).cpu().detach().numpy()
    # 2.2 获取已知数据的真实置信度,已有truePro_observe
    # 2.3 获取准确率
    truePro_acc_dis = get_truePro_acc(truePro_dis, truePro_observe, continuous_cols, data_m)

    # # 3. 查看当前cell置信度结果的准确率
    # 3.1 获取vae的得到的结果
    decoder_z_impute = zero_feed_data + generate_x * (1 - M_tensor)
    truePro = true_Pro(decoder_z_impute, fields, value_cat, values, miss_data_x, data_m, enc, ori_data)
    truePro[np.where(data_m == 1)] = 1
    truePro[:, continuous_cols] = 0.
    # acc = get_truePro_acc(cell_acc, truePro, continuous_cols, data_m)

    # 4. 获取每个单元格的准确率
    vae_cell_acc = truePro_acc_vae * get_trueProVAE(decoder_z_impute, fields)
    # 4.1 获取dis得到的结果
    truePro_dis = discriminator(decoder_z_impute).cpu().detach().numpy()
    truePro_dis[:, continuous_cols] = 0
    dis_cell_acc = truePro_acc_dis * truePro_dis
    # 4.2 加权平均，并把observer处的置信度置为1
    cell_acc = (vae_cell_acc + dis_cell_acc + truePro) / (truePro_acc_dis + truePro_acc_vae + 1)
    # cell_acc = truePro
    cell_acc[np.where(data_m == 1)] = 1
    cell_acc[:, continuous_cols] = truePro_dis[:, continuous_cols]
    return cell_acc

def get_trueProVAE(decodes, fields):
    truePro = torch.zeros(decodes.shape[0], len(fields))
    cur_index = 0
    for index, field in enumerate(fields):
        if field.data_type == "Categorical Data":
            dim = field.dim()
            data = decodes[:, cur_index:cur_index+dim]
            z_scores = (data - data.mean(dim=1, keepdim=True)) / data.std(dim=1, keepdim=True)
            sigmoid_results = data
            max_values, _ = torch.max(sigmoid_results, dim=1, keepdim=True)
            truePro[:, index] = max_values.squeeze(1)
            cur_index = cur_index + dim
        else:
            cur_index = cur_index + 1
    return truePro.cpu().detach().numpy()

def update_impute_code(decoder_z_impute, best_impute_code,fields,cell_acc, impute_data_acc):
    begin_list = [0]
    begin = 0
    for index, field in enumerate(fields):
        if field.data_type == "Categorical Data":
            begin += len(field.dict)
        else:
            begin += 1
        begin_list.append(begin)
    for row, row_acc in enumerate(cell_acc):
        for col, col_acc in enumerate(row_acc):
            if col_acc > impute_data_acc[row, col]:
                best_impute_code[row, begin_list[col]:begin_list[col+1]] = decoder_z_impute[row, begin_list[col]:begin_list[col+1]]
                impute_data_acc[row, col] = col_acc
    return

def VAE_Dis_train(use_discriminator, vae, discriminator,epochs, steps_per_epoch, batch_size, loss_weight, data_m, impute_data_code,label_data, fields, value_cat, values,miss_data_x,enc,ori_data,continuous_cols,label_num,device,data_name, val_data):
    # 根据data_m获取
    print("--------------VAE train---------------------")
    torch.manual_seed(3047)
    if device == torch.device('cuda:0'):
        discriminator.cuda()
        vae.cuda()
    M_tensor = get_M_by_data_m(data_m, fields, device)
    zero_feed_data = M_tensor * impute_data_code
    # 根据data_m,获取true_data用来做验证集
    if label_num > 0:
        valid_data_index = get_valid_data_index(data_m, discriminator, impute_data_code, device)
        train_data_index = [i for i in range(len(data_m)) if i not in valid_data_index]
        valid_data_code = zero_feed_data[valid_data_index]
        label_data_code = torch.FloatTensor(label_data.values).to(device)
    optimizer_vae = optim.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)
    x = impute_data_code.to(device)

    best_impute_code = x.clone()
    if label_num > 0:
        x_valid = x[valid_data_index]
        y_valid = label_data_code[valid_data_index]
    m_data = torch.tensor(data_m).float().to(device)
    discriminator.train()
    vae.train()
    rmse_list = []
    discriminator_loss_list = []
    generate_loss_list = []
    ARMSE_list, AMAE_list, Acc_list = [], [], []
    Acc_max = 0
    # first_rmse, mae = test_impute_data_rmse(x, fields, value_cat, values, miss_data_x.copy(), data_m, enc, ori_data,
    #                                         continuous_cols)
    # ARMSE_list.append(first_rmse)
    # AMAE_list.append(mae)
    # new_FDs_model_list = FDs_model_list.copy()
    for epoch in tqdm(range(epochs)):
        # sample一些x
        x_attention_sample, sample_data_index = sample_x(x, batch_size)
        M_sample = M_tensor[sample_data_index]
        m_sample = data_m[sample_data_index]
        zero_feed_sample = zero_feed_data[sample_data_index]
        m_data_sample = m_data[sample_data_index]
        if label_num>0:
            y_train = label_data_code[sample_data_index]
        discriminator_loss = 0
        if use_discriminator:
            for it in range(steps_per_epoch):
                # 优化discriminator
                z_, mu, log_var, generate_x, generate_out_x = vae(x_attention_sample)
                decoder_z_impute = zero_feed_sample + (1 - M_sample) * generate_out_x
                discriminator_loss = train_discriminator(optimizer_D, decoder_z_impute, discriminator, x_attention_sample, zero_feed_sample,M_sample,m_data_sample,loss_weight)
                discriminator_loss_list.append(discriminator_loss.item())
        for it in range(steps_per_epoch):
            if epoch % 100 == 0 and label_num > 0:
                z_, mu, log_var, generate_x, generate_out_x = vae(x)
                decoder_z_impute = zero_feed_data + (1 - M_tensor) * generate_x
                x_train_code = decoder_z_impute[train_data_index]
                y_train_code = label_data_code[train_data_index]
                L_loss, Acc = train_L_code(1000, x_train_code, y_train_code, x_valid, y_valid, label_num, device)
                if Acc > Acc_max:
                    Acc_max = Acc
                    torch.save(vae.state_dict(), 'vae.pth')
            else:
                L_loss = 0
            generator_loss, Reconstruction_loss, KL_loss, Generator_Discriminator_z_loss, L_loss = train_vae(use_discriminator, epoch, loss_weight, optimizer_vae, x_attention_sample, zero_feed_sample, M_sample, m_data_sample, fields, L_loss, vae, discriminator)
            generate_loss_list.append(generator_loss)
        if epoch % 50 == 0 and epoch > 0:
            vae.eval()
            discriminator.eval()
            z_, mu, log_var, generate_x, generate_out_x = vae(x)
            # cell_acc = get_cell_true(generate_x, zero_feed_data, fields, data_m, continuous_cols, discriminator,generate_out_x, M_tensor, value_cat, values, miss_data_x.copy(), enc, ori_data, device)
            # 根据cell_acc
            # new_FDs_model_list = update_FD_models(generate_x, zero_feed_data, fields, data_m, M_tensor, value_cat, values, miss_data_x.copy(), enc, device, eq_dict, FDs_model_list, cell_acc, continuous_cols)
            code = zero_feed_data + (1 - M_tensor) * generate_x
            # truePro_generate = true_Pro(generate_out_x,fields,value_cat,values,miss_data_x,data_m,enc,ori_data,continuous_cols)
            # 根据填充最后的数据的准确率，使用准确率最大的数据进行填充,更爱best_impute_code
            # update_impute_code(decoder_z_impute, best_impute_code,fields,cell_acc, impute_data_acc)
            rmse, mae = test_impute_data_rmse(code,fields,value_cat,values,miss_data_x.copy(),data_m,enc,ori_data,continuous_cols)
            impute_data = reconvert_data(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc)
            down_load_acc = get_down_acc(impute_data, label_data, val_data, label_num, device, value_cat, continuous_cols, enc)
            # if len(FDs_model_list) != 0:
            #     FD_loss = FD_loss.item()
            # if label_num > 0:
            #     acc = test_impute_data_acc(decoder_z_impute,valid_data_index,label_data_code,train_data_index,label_num,device)
            #     print("ARMSE为：{}    AMAE为：{}    Vae_loss:{}  Res_loss:{}  G_D_loss:{}     D_loss:{}    Learner_ACC:{}    kl_loss:{}".format(
            #             rmse, mae, generator_loss,
            #             Reconstruction_loss, Generator_Discriminator_z_loss, discriminator_loss,
            #             acc, KL_loss))
            # else:
            #     print("ARMSE为：{}   AMAE为：{}     Vae_loss:{}  Res_loss:{}  G_D_loss:{}     D_loss:{}   kl_loss:{}".format(
            #             rmse,mae, generator_loss, Reconstruction_loss, Generator_Discriminator_z_loss, discriminator_loss,
            #             KL_loss))
            print("RMSE为：{:.3f}, MAE为：{:.3f}, Acc为：{:.3f}".format(rmse,mae,down_load_acc))
            ARMSE_list.append(rmse)
            AMAE_list.append(mae)
            Acc_list.append(down_load_acc)
            # min_index = ARMSE_list.index(min(ARMSE_list))
    return min(ARMSE_list), min(AMAE_list), max(Acc_list)
    # FD_list = []
    # for new_FD in new_FDs_model_list:
    #     x_index_list = new_FD.x_index_list
    #     attr_list = []
    #     for x_index in x_index_list:
    #         attr_list.append(values[x_index])
    #     print("LHS部分的属性为：{}  ---->  {}".format(attr_list, values[new_FD.y_index]))
    # if label_num > 0 and Acc_max <= first_acc:
    #     print("准确率最大时RMSE为：{}   Acc为：{}\n".format(first_rmse, first_acc))
    # elif label_num > 0:
    #     vae.load_state_dict(torch.load('vae.pth'))
    #     vae.eval()
    #     z_, mu, log_var, generate_x, generate_out_x = vae(x)
    #     decoder_z_impute = impute_data_code * M_tensor + (1 - M_tensor) * generate_x
    #     rmse, mse = test_impute_data_rmse(decoder_z_impute, fields, value_cat, values, miss_data_x, data_m, enc, ori_data,
    #                                  continuous_cols)
    #     acc = test_impute_data_acc(decoder_z_impute, valid_data_index, label_data_code, train_data_index, label_num, device)
    #     print("准确率最大时RMSE为：{}   Acc为：{}\n".format(rmse, Acc_max))
    # else:
    #     vae.eval()
    #     z_, mu, log_var, generate_x, generate_out_x = vae(x)
    #     decoder_z_impute = impute_data_code * M_tensor + (1 - M_tensor) * generate_x
    #     rmse,mse= test_impute_data_rmse(decoder_z_impute, fields, value_cat, values, miss_data_x, data_m, enc, ori_data,continuous_cols)
    #     print("准确率最大时RMSE为：{}\n".format(rmse))
    #
    # plt.figure(1)
    # epoch_list = list(np.arange(len(rmse_list)))
    # plt.plot(epoch_list, rmse_list)
    # plt.title("RMSE")
    # # add a label to the x-axis
    # plt.xlabel("Epoch")
    # # add a label to the y-axis
    # plt.ylabel("Loss")
    # # show the plot
    # plt.show()
    #
    #
    # plt.figure(2)
    # epoch_list = list(np.arange(len(discriminator_loss_list)))
    # plt.plot(epoch_list, discriminator_loss_list)
    # plt.title("D_loss")
    # # add a label to the x-axis
    # plt.xlabel("Epoch")
    # # add a label to the y-axis
    # plt.ylabel("D_loss")
    # # show the plot
    # plt.show()
