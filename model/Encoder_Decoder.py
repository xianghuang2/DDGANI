import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.nn as nn
import torch.nn.functional as fun
import math
from model.FD_model import get_my_FD_loss
import matplotlib.pyplot as plt
import torch
from model.Discriminator_model import D, train_discriminator
import torch.optim as optim
import numpy as np
from utils.util import get_M_by_data_m, test_impute_data_rmse, test_impute_data_acc,get_valid_data_index,sample_x
from model.Learner import train_L_code


class Encoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, encoder_out_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(encoder_dim, encoder_out_dim),
            nn.BatchNorm1d(encoder_out_dim),
            nn.ReLU(inplace=False)
        )

        self.en_to_la = nn.Linear(encoder_out_dim, latent_dim)
        self.mu = nn.Linear(encoder_out_dim, latent_dim)  # mean
        self.sigma = nn.Linear(encoder_out_dim, latent_dim)  # sigma,z为浅层的维度
        self.z_linear = nn.Linear(latent_dim, latent_dim)
        self.softmax = nn.Softmax()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)


    def parameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(mu.size(0), mu.size(1)).to(std.device)
        z = mu + eps * std
        return z

    def self_attention(self, x):
        corr_map = self.corr_map.to(x.device)
        corr_map = self.en_to_la(self.encoder(corr_map))
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        value_list = []
        if x.size()[0] > 1000:
            x = x.cpu()
            query = query.cpu()
            key = key.cpu()
            value = value.cpu()
            corr_map = corr_map.cpu()
        for index, i in enumerate(self.filed):
            cur_col_corr = corr_map[index]
            cur_col_corr = cur_col_corr.repeat(query.shape[0], 1)
            corr_cur = torch.matmul(query * cur_col_corr, key.T)
            corr_cur.diagonal(offset=0).fill_(float('-inf'))
            corr_cur = torch.softmax(corr_cur, dim=1)
            # k = int(corr_cur.shape[0] * 0.05)
            # top_tensor = torch.topk(corr_cur, k=k, dim=1).values[:, -1].unsqueeze(1)
            # top_tensor = top_tensor.expand_as(corr_cur)
            # corr_cur[corr_cur < top_tensor] = 0
            # row_sum = torch.sum(corr_cur, dim=1)
            # corr_cur = corr_cur / row_sum.unsqueeze(0)
            x_new = torch.matmul(corr_cur, value)
            value_list.append(x_new)

        z_ = torch.stack(value_list, dim=0)     # m * n * latent_dim
        # query = query.repeat((self.num_headers, 1, 1))
        # key = key.repeat((self.num_headers, 1, 1))
        # value = value.repeat((self.num_headers, 1, 1))
        #
        # attention_score = torch.bmm(query, key.transpose(-2, -1)) * self.scale_factor

        # attention_prob = torch.softmax(attention_score, dim=-1)
        # context = torch.bmm(attention_prob, value)
        # context = torch.reshape(context, (context.shape[1], -1)).float()
        # output = self.output_linear(context)
        return z_.to(self.device)

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
        out1 = self.encoder(x)
        z = self.en_to_la(out1)
        # z_ = self.z_linear(z_)
        # z_ = self.self_attention(z_)
        return z


class Decoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, encoder_out_dim, latent_dim):
        super().__init__()

        self.input_dim = input_dim

        self.latent_dim = latent_dim

        self.encoder_out_dim = encoder_out_dim
        self.linear_1 = nn.Linear(latent_dim, encoder_out_dim)
        self.linear_2 = nn.Linear(encoder_out_dim, encoder_dim)
        self.linear_3 = nn.Linear(encoder_dim, input_dim)
        self.batch_normal1 = nn.BatchNorm1d(encoder_out_dim)
        self.batch_normal2 = nn.BatchNorm1d(encoder_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.generate = nn.Sequential(
            nn.Linear(latent_dim, encoder_out_dim),
            # nn.BatchNorm1d(encoder_out_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(encoder_out_dim, encoder_dim),
            # nn.BatchNorm1d(decoder_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(encoder_dim, input_dim),
            nn.Sigmoid()
        )



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
        out = self.linear_1(x)
        if x.dim() == 2:
            out = self.batch_normal1(out)
        else:
            m, n, k = out.size()
            matrix = out.transpose(1, 2).contiguous().view(-1, k)
            output_matrix = self.batch_normal1(matrix)
            out = output_matrix.view(m, n, k)
        out = self.dropout(self.relu(out))
        out = self.linear_2(out)
        if x.dim() == 2:
            out = self.batch_normal2(out)
        else:
            m, n, k = out.size()
            matrix = out.transpose(1, 2).contiguous().view(-1, k)
            output_matrix = self.batch_normal2(matrix)
            out = output_matrix.view(m, n, k)
        out = self.dropout(self.relu(out))
        out = self.linear_3(out)
        # out = self.tanh(out)
        # 对类别数据做softmax
        current_ind = 0
        decodes = []
        out_decoders = []
        if x.dim() == 2:
            for i in range(len(self.fields)):
                if self.fields[i].data_type == "Categorical Data":
                    dim = self.fields[i].dim()  # one-hot向量维度
                    data = fun.softmax(out[:, current_ind:current_ind + dim], dim=1)
                    decodes.append(data)
                    max_index = torch.argmax(data, dim=1)
                    one_hot = torch.zeros_like(data)
                    one_hot[:, max_index] = 1
                    out_decoders.append(one_hot)
                    current_ind = current_ind + dim
                else:
                    decodes.append(out[:, current_ind:current_ind + 1])
                    out_decoders.append(out[:, current_ind:current_ind + 1])
                    current_ind = current_ind + 1
            decodes = torch.cat(decodes, dim=1)
            out_decoders = torch.cat(out_decoders, dim=1)
        else:
            for i in range(len(self.fields)):
                if self.fields[i].data_type == "Categorical Data":
                    dim = self.fields[i].dim()  # one-hot向量维度
                    data = fun.softmax(out[i, :, current_ind:current_ind + dim], dim=1)
                    decodes.append(data)
                    max_index = torch.argmax(data, dim=1)
                    one_hot = torch.zeros_like(data)
                    one_hot[:, max_index] = 1
                    out_decoders.append(one_hot)
                    current_ind = current_ind + dim
                else:
                    decodes.append(out[i, :, current_ind:current_ind + 1])
                    out_decoders.append(out[i, :, current_ind:current_ind + 1])
                    current_ind = current_ind + 1
            decodes = torch.cat(decodes, dim=1)
            out_decoders = torch.cat(out_decoders, dim=1)
        return decodes, out_decoders


class Encoder_Decoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, encoder_out_dim, latent_dim, fields):
        super().__init__()
        self.fields = fields
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(encoder_dim, encoder_out_dim),
            nn.BatchNorm1d(encoder_out_dim),
            nn.ReLU(inplace=False)
        )

        self.en_to_la = nn.Linear(encoder_out_dim, latent_dim)
        self.mu = nn.Linear(encoder_out_dim, latent_dim)  # mean
        self.sigma = nn.Linear(encoder_out_dim, latent_dim)  # sigma,z为浅层的维度
        self.z_linear = nn.Linear(latent_dim, latent_dim)
        self.softmax = nn.Softmax()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)

        self.input_dim = input_dim

        self.latent_dim = latent_dim

        self.encoder_out_dim = encoder_out_dim
        self.linear_1 = nn.Linear(latent_dim, encoder_out_dim)
        self.linear_2 = nn.Linear(encoder_out_dim, encoder_dim)
        self.linear_3 = nn.Linear(encoder_dim, input_dim)
        self.batch_normal1 = nn.BatchNorm1d(encoder_out_dim)
        self.batch_normal2 = nn.BatchNorm1d(encoder_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.generate = nn.Sequential(
            nn.Linear(latent_dim, encoder_out_dim),
            # nn.BatchNorm1d(encoder_out_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(encoder_out_dim, encoder_dim),
            # nn.BatchNorm1d(decoder_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(encoder_dim, input_dim),
            nn.Sigmoid()
        )


    def parameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(mu.size(0), mu.size(1)).to(std.device)
        z = mu + eps * std
        return z

    def self_attention(self, x):
        corr_map = self.corr_map.to(x.device)
        corr_map = self.en_to_la(self.encoder(corr_map))
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        value_list = []
        if x.size()[0] > 1000:
            x = x.cpu()
            query = query.cpu()
            key = key.cpu()
            value = value.cpu()
            corr_map = corr_map.cpu()
        for index, i in enumerate(self.filed):
            cur_col_corr = corr_map[index]
            cur_col_corr = cur_col_corr.repeat(query.shape[0], 1)
            corr_cur = torch.matmul(query * cur_col_corr, key.T)
            corr_cur.diagonal(offset=0).fill_(float('-inf'))
            corr_cur = torch.softmax(corr_cur, dim=1)
            # k = int(corr_cur.shape[0] * 0.05)
            # top_tensor = torch.topk(corr_cur, k=k, dim=1).values[:, -1].unsqueeze(1)
            # top_tensor = top_tensor.expand_as(corr_cur)
            # corr_cur[corr_cur < top_tensor] = 0
            # row_sum = torch.sum(corr_cur, dim=1)
            # corr_cur = corr_cur / row_sum.unsqueeze(0)
            x_new = torch.matmul(corr_cur, value)
            value_list.append(x_new)

        z_ = torch.stack(value_list, dim=0)     # m * n * latent_dim
        # query = query.repeat((self.num_headers, 1, 1))
        # key = key.repeat((self.num_headers, 1, 1))
        # value = value.repeat((self.num_headers, 1, 1))
        #
        # attention_score = torch.bmm(query, key.transpose(-2, -1)) * self.scale_factor

        # attention_prob = torch.softmax(attention_score, dim=-1)
        # context = torch.bmm(attention_prob, value)
        # context = torch.reshape(context, (context.shape[1], -1)).float()
        # output = self.output_linear(context)
        return z_.to(self.device)

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
        out1 = self.encoder(x)
        z = self.en_to_la(out1)
        out = self.linear_1(z)
        if z.dim() == 2:
            out = self.batch_normal1(out)
        else:
            m, n, k = out.size()
            matrix = out.transpose(1, 2).contiguous().view(-1, k)
            output_matrix = self.batch_normal1(matrix)
            out = output_matrix.view(m, n, k)
        out = self.dropout(self.relu(out))
        out = self.linear_2(out)
        if z.dim() == 2:
            out = self.batch_normal2(out)
        else:
            m, n, k = out.size()
            matrix = out.transpose(1, 2).contiguous().view(-1, k)
            output_matrix = self.batch_normal2(matrix)
            out = output_matrix.view(m, n, k)
        out = self.dropout(self.relu(out))
        out = self.linear_3(out)
        # out = self.tanh(out)
        # 对类别数据做softmax
        current_ind = 0
        decodes = []
        out_decoders = []
        if z.dim() == 2:
            for i in range(len(self.fields)):
                if self.fields[i].data_type == "Categorical Data":
                    dim = self.fields[i].dim()  # one-hot向量维度
                    data = fun.softmax(out[:, current_ind:current_ind + dim], dim=1)
                    decodes.append(data)
                    max_index = torch.argmax(data, dim=1)
                    one_hot = torch.zeros_like(data)
                    one_hot[:, max_index] = 1
                    out_decoders.append(one_hot)
                    current_ind = current_ind + dim
                else:
                    decodes.append(out[:, current_ind:current_ind + 1])
                    out_decoders.append(out[:, current_ind:current_ind + 1])
                    current_ind = current_ind + 1
            decodes = torch.cat(decodes, dim=1)
            out_decoders = torch.cat(out_decoders, dim=1)
        else:
            for i in range(len(self.fields)):
                if self.fields[i].data_type == "Categorical Data":
                    dim = self.fields[i].dim()  # one-hot向量维度
                    data = fun.softmax(out[i, :, current_ind:current_ind + dim], dim=1)
                    decodes.append(data)
                    max_index = torch.argmax(data, dim=1)
                    one_hot = torch.zeros_like(data)
                    one_hot[:, max_index] = 1
                    out_decoders.append(one_hot)
                    current_ind = current_ind + dim
                else:
                    decodes.append(out[i, :, current_ind:current_ind + 1])
                    out_decoders.append(out[i, :, current_ind:current_ind + 1])
                    current_ind = current_ind + 1
            decodes = torch.cat(decodes, dim=1)
            out_decoders = torch.cat(out_decoders, dim=1)
        return decodes, out_decoders




def E_D_loss_func(fields, reconstruct, x, M):
    batch_size = x.size(0)
    MSE = 0
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
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pre, label)
            BCE += loss
            BCE_num = BCE_num + 1
        curr += dim
    if BCE_num == 0:
        BCE_num = BCE_num + 1
    if MSE_num == 0:
        MSE_num = MSE_num + 1
    if math.isnan((MSE / MSE_num + BCE / BCE_num) / 2):
        print(1)
    # 对MSE和BCE进行权衡
    return MSE/MSE_num , BCE/BCE_num


def train_E_D(use_discriminator, loss_weight, optimizer, x, zero_feed_code, M_tensor, m_data, fields,L_loss, E_D, discriminator, FDs_model_list):
    optimizer.zero_grad()
    generate_x, generate_out_x = E_D(x)
    decoder_z_impute = zero_feed_code + (1 - M_tensor) * generate_x
    MSE_loss, BCE_loss= E_D_loss_func(fields, generate_x, x, M_tensor)
    if BCE_loss != 0:
        BCE_loss = BCE_loss * loss_weight['BCE_weight']
    Reconstruction_loss = MSE_loss + BCE_loss
    FD_loss = get_my_FD_loss(FDs_model_list, decoder_z_impute, fields)
    if use_discriminator:
        discriminator_z = discriminator(decoder_z_impute)
        Generator_Discriminator_z_loss = (-torch.mean((1 - m_data) * torch.log(discriminator_z + 1e-8)))
    else:
        Generator_Discriminator_z_loss = 0
    if L_loss == 0:
        generator_loss = loss_weight[
                             'Res_weight'] * Reconstruction_loss + Generator_Discriminator_z_loss * \
                         loss_weight['G_weight'] + FD_loss * loss_weight['FD_weight']
    else:
        generator_loss = loss_weight['Res_weight'] * Reconstruction_loss + Generator_Discriminator_z_loss * \
                         loss_weight['G_weight'] + FD_loss * loss_weight['FD_weight'] + L_loss * loss_weight['L_weight']
    generator_loss.backward()
    optimizer.step()
    if use_discriminator:
        return generator_loss.item(),Reconstruction_loss.item(),Generator_Discriminator_z_loss.item(),FD_loss,L_loss
    else:
        return generator_loss.item(),Reconstruction_loss.item(),Generator_Discriminator_z_loss,FD_loss,L_loss


def E_D_Dis_train(use_discriminator, E_D, discriminator,FDs_model_list, epochs, steps_per_epoch, batch_size, loss_weight, data_m, impute_data_code, label_data, fields, value_cat, values,miss_data_x,enc,ori_data,continuous_cols,label_num,device):
    # 根据data_m获取
    print("--------------E_D train---------------------")
    torch.manual_seed(3047)
    if device == torch.device('cuda:0'):
        discriminator.cuda()
        E_D.cuda()
    M_tensor = get_M_by_data_m(data_m, fields, device)
    zero_feed_data = M_tensor * impute_data_code
    # 根据data_m,获取true_data用来做验证集
    if label_num > 0:
        valid_data_index = get_valid_data_index(data_m, discriminator, impute_data_code, device)
        train_data_index = [i for i in range(len(data_m)) if i not in valid_data_index]
        valid_data_code = zero_feed_data[valid_data_index]
        label_data_code = torch.FloatTensor(label_data.values).to(device)

    # gain_parameters = {'batch_size': 128,
    #                  'hint_rate': 0.9,
    #                  'alpha': 100,
    #                  'iterations': 10000}
    # imputed_data_x, impute_data_x_code = gain(nan_data, gain_parameters, ori_data)
    # # imputed_data_x, impute_data_x_code = mean(nan_data)
    # x_valid = torch.tensor(impute_data_x_code[valid_data_index],dtype=torch.float).to(device)
    # y_valid = label_data_code[valid_data_index]
    # x_train = torch.tensor(impute_data_x_code[train_data_index],dtype=torch.float).to(device)
    # y_train = label_data_code[train_data_index]
    # imputed_data_x = pd.DataFrame(imputed_data_x)
    # rmse, mse = errorLoss(imputed_data_x, ori_data, data_m, value_cat, continuous_cols, enc)
    # L_loss, Acc = train_L_code(1000, x_train, y_train, x_valid, y_valid, label_num, device)
    # print("Acc为：{}  RMSE：{}".format(Acc,rmse))

    # lr表示学习率，betas表示Adam算法中的动量参数，eps表示数值稳定性参数，weight_decay表示L2正则化参数。具体来说，betas参数控制了梯度的一阶矩和二阶矩的衰减率，eps参数用于防止除以零的情况发生，weight_decay参数用于控制L2正则化的强度。
    optimizer_vae = optim.Adam(E_D.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
    # # x为原有数据+attention数据
    # x = zero_feed_data + x * (1 - M_tensor)
    x = impute_data_code.to(device)
    if label_num > 0:
        x_valid = x[valid_data_index]
        y_valid = label_data_code[valid_data_index]
    m_data = torch.tensor(data_m).float().to(device)
    discriminator.train()
    E_D.train()
    rmse_list = []
    discriminator_loss_list = []
    generate_loss_list = []
    first_rmse = test_impute_data_rmse(x,fields,value_cat,values,miss_data_x.copy(),data_m,enc,ori_data,continuous_cols)
    if label_num > 0:
        first_acc = test_impute_data_acc(x,valid_data_index,label_data_code,train_data_index,label_num,device)
        print("初始填充数据的RMSE为：{}，ACC为{}".format(first_rmse, first_acc))
    else:
        print("初始填充数据的RMSE为：{}".format(first_rmse))
    Acc_max = 0
    for epoch in range(epochs):
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
            for it in range(int(steps_per_epoch)):
                # 优化discriminator
                generate_x, generate_out_x = E_D(x_attention_sample)
                discriminator_loss = train_discriminator(optimizer_D, generate_x, discriminator, x_attention_sample, zero_feed_sample,M_sample,m_data_sample,loss_weight)
                discriminator_loss_list.append(discriminator_loss.item())

        for it in range(steps_per_epoch):
            if epoch % 100 == 0 and label_num > 0:
                generate_x, generate_out_x = E_D(x)
                decoder_z_impute = zero_feed_data + (1 - M_tensor) * generate_x
                x_train_code = decoder_z_impute[train_data_index]
                y_train_code = label_data_code[train_data_index]
                L_loss, Acc = train_L_code(1000, x_train_code, y_train_code, x_valid, y_valid, label_num, device)
                if Acc > Acc_max:
                    Acc_max = Acc
                    torch.save(E_D.state_dict(), 'E_D.pth')
            else:
                L_loss = 0
            generator_loss, Reconstruction_loss, Generator_Discriminator_z_loss, FD_loss, L_loss = train_E_D(use_discriminator, loss_weight, optimizer_vae, x_attention_sample, zero_feed_sample, M_sample, m_data_sample, fields, L_loss, E_D, discriminator, FDs_model_list)
            generate_loss_list.append(generator_loss)
        if epoch % 100 == 0:
            E_D.eval()
            generate_x, generate_out_x = E_D(x)
            decoder_z_impute = zero_feed_data + (1 - M_tensor) * generate_x
            # 填充最后的数据
            rmse = test_impute_data_rmse(decoder_z_impute,fields,value_cat,values,miss_data_x.copy(),data_m,enc,ori_data,continuous_cols)
            if len(FDs_model_list) != 0:
                FD_loss = FD_loss.item()
            if label_num > 0:
                acc = test_impute_data_acc(decoder_z_impute,valid_data_index,label_data_code,train_data_index,label_num,device)
                print("RMSE为：{}    Vae_loss:{}  Res_loss:{}  G_D_loss:{}     D_loss:{}    Learner_ACC:{}   FD_loss:{}".format(
                        rmse, generator_loss,
                        Reconstruction_loss, Generator_Discriminator_z_loss, discriminator_loss,
                        acc, FD_loss))
            else:
                print("RMSE为：{}    Vae_loss:{}  Res_loss:{}  G_D_loss:{}     D_loss:{}   FD_loss:{}".format(
                        rmse, generator_loss, Reconstruction_loss, Generator_Discriminator_z_loss, discriminator_loss,
                       FD_loss))
            rmse_list.append(rmse)
    print("--------最小的RMSE: {}--------\n".format((min(rmse_list))))

    if label_num > 0 and Acc_max <= first_acc:
        print("准确率最大时RMSE为：{}   Acc为：{}\n".format(first_rmse, first_acc))
    elif label_num > 0:
        E_D.load_state_dict(torch.load('E_D.pth'))
        E_D.eval()
        generate_x, generate_out_x = E_D(x)
        decoder_z_impute = impute_data_code * M_tensor + (1 - M_tensor) * generate_x
        rmse = test_impute_data_rmse(decoder_z_impute, fields, value_cat, values, miss_data_x.copy(), data_m, enc, ori_data,
                                     continuous_cols)
        acc = test_impute_data_acc(decoder_z_impute, valid_data_index, label_data_code, train_data_index, label_num, device)
        print("准确率最大时RMSE为：{}   Acc为：{}\n".format(rmse, Acc_max))
    else:
        E_D.eval()
        generate_x, generate_out_x = E_D(x)
        decoder_z_impute = impute_data_code * M_tensor + (1 - M_tensor) * generate_x
        rmse = test_impute_data_rmse(decoder_z_impute, fields, value_cat, values, miss_data_x.copy(), data_m, enc, ori_data,
                                     continuous_cols)
        print("准确率最大时RMSE为：{}\n".format(rmse))

    plt.figure(1)
    epoch_list = list(np.arange(len(rmse_list)))
    plt.plot(epoch_list, rmse_list)
    plt.title("RMSE")
    # add a label to the x-axis
    plt.xlabel("Epoch")
    # add a label to the y-axis
    plt.ylabel("Loss")
    # show the plot
    plt.show()


    plt.figure(4)
    epoch_list = list(np.arange(len(discriminator_loss_list)))
    plt.plot(epoch_list, discriminator_loss_list)
    plt.title("D_loss")
    # add a label to the x-axis
    plt.xlabel("Epoch")
    # add a label to the y-axis
    plt.ylabel("D_loss")
    # show the plot
    plt.show()

