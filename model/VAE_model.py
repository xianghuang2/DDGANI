import pandas as pd
import torch.nn as nn
import torch.nn.functional as fun
import math

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
        self.sigma = nn.Linear(encoder_out_dim, latent_dim)  # sigma,z as latent dim
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
        # get z_ attention
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
        # softmax for cat
        current_ind = 0
        decodes = []
        out_decodes = []
        if x.dim() == 2:
            for i in range(len(self.fields)):
                if self.fields[i].data_type == "Categorical Data":
                    dim = self.fields[i].dim()  # one-hot dim
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
                    dim = self.fields[i].dim()  # one-hot dim
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
    KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
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
            loss_fn = nn.MSELoss()
            mse = loss_fn(pre, label)
            MSE += mse
            MSE_num = MSE_num + 1
        else:
            m = M[:, curr:curr + dim].cpu().detach().numpy()
            good_rows = np.where(np.all(m == 1, axis=1))[0]
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
    if use_discrinminator:
        discriminator_z = discriminator(decoder_z_impute)
        Generator_Discriminator_z_loss = (-torch.mean((1 - m_data) * torch.log(discriminator_z + 1e-8)))
    else:
        Generator_Discriminator_z_loss = 0
    generator_loss =  loss_weight['VAE_Res_weight'] * Reconstruction_loss + kl_weight * KL_loss + Generator_Discriminator_z_loss * \
                         loss_weight['VAE_G_weight'] + L_loss * loss_weight['L_weight']
    generator_loss.backward()
    optimizer.step()
    if use_discrinminator:
        return generator_loss.item(),Reconstruction_loss.item(),(kl_weight * KL_loss).item(),Generator_Discriminator_z_loss.item(),L_loss
    else:
        return generator_loss.item(),Reconstruction_loss.item(),(kl_weight * KL_loss).item(),Generator_Discriminator_z_loss,L_loss


def VAE_Dis_train(use_discriminator, vae, discriminator,epochs, steps_per_epoch, batch_size, loss_weight, data_m, impute_data_code,label_data, fields, value_cat, values,miss_data_x,enc,ori_data,continuous_cols,label_num,device):

    torch.manual_seed(3047)
    if device == torch.device('cuda:0'):
        discriminator.cuda()
        vae.cuda()
    M_tensor = get_M_by_data_m(data_m, fields, device)
    zero_feed_data = M_tensor * impute_data_code

    valid_data_index = [i for i in range(len(data_m)) if all(val == 1 for val in data_m[i])]
    train_data_index = [i for i in range(len(data_m)) if i not in valid_data_index]
    label_data_code = torch.FloatTensor(label_data.values).to(device)
    if len(valid_data_index) > 0:
        valid_data_code = zero_feed_data[valid_data_index]

    optimizer_vae = optim.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)
    x = impute_data_code.to(device)

    if len(valid_data_index) > 0:
        x_valid = x[valid_data_index]
        y_valid = label_data_code[valid_data_index]
    m_data = torch.tensor(data_m).float().to(device)
    discriminator.train()
    vae.train()
    rmse_max = 9999999
    for epoch in tqdm(range(epochs)):
        # sample x
        x_attention_sample, sample_data_index = sample_x(x, batch_size)
        M_sample = M_tensor[sample_data_index]
        m_sample = data_m[sample_data_index]
        zero_feed_sample = zero_feed_data[sample_data_index]
        m_data_sample = m_data[sample_data_index]
        if len(valid_data_index)>0:
            y_train = label_data_code[sample_data_index]
        discriminator_loss = 0
        if use_discriminator:
            for it in range(steps_per_epoch):
                # backward discriminator
                z_, mu, log_var, generate_x, generate_out_x = vae(x_attention_sample)
                decoder_z_impute = zero_feed_sample + (1 - M_sample) * generate_out_x
                discriminator_loss = train_discriminator(optimizer_D, decoder_z_impute, discriminator, x_attention_sample, zero_feed_sample,M_sample,m_data_sample,loss_weight)
        for it in range(steps_per_epoch):
            if epoch % 100 == 0 and len(valid_data_index) > 0:
                z_, mu, log_var, generate_x, generate_out_x = vae(x)
                decoder_z_impute = zero_feed_data + (1 - M_tensor) * generate_x
                x_train_code = decoder_z_impute[train_data_index]
                y_train_code = label_data_code[train_data_index]
                L_loss, Acc = train_L_code(1000, x_train_code, y_train_code, x_valid, y_valid, label_num, device)
            else:
                L_loss = 0
            generator_loss, Reconstruction_loss, KL_loss, Generator_Discriminator_z_loss, L_loss = train_vae(use_discriminator, epoch, loss_weight, optimizer_vae, x_attention_sample, zero_feed_sample, M_sample, m_data_sample, fields, L_loss, vae, discriminator)
        if epoch % 100 == 0 and epoch > 0:
            vae.eval()
            discriminator.eval()
            z_, mu, log_var, generate_x, generate_out_x = vae(x)
            code = zero_feed_data + (1 - M_tensor) * generate_x
            rmse, mae = test_impute_data_rmse(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc,
                                              ori_data, continuous_cols)
            if rmse < rmse_max:
                impute_data = reconvert_data(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc)
                rmse_max = rmse
    return impute_data.values

