from abc import abstractmethod
import time

import pandas as pd
from torch.cuda.amp import GradScaler
from torch.nn import init
import math
import torch.nn as nn
from torchmetrics import MeanMetric
from tqdm import tqdm
from torch.cuda import amp
import torch
import torch.optim as optim
import numpy as np
from BaseLine.Mean import fill_data_mean
from model.Diffusion import Diffusion_FD_loss, G_d_loss_func, res_loss_func, train_one_epoch_Generator
from model.Discriminator_model import train_discriminator
from model.FD_model import get_my_FD_loss, update_FD_models, get_eq_dict, get_trueProObserve
from utils.util import get_M_by_data_m, test_impute_data_rmse, test_impute_data_acc, get_valid_data_index, sample_x, \
    reconvert_data, get_down_acc, test_fd_data_rmse, test_impute_data_Acc
from model.Learner import train_L_code
from torch.utils.data import Dataset, DataLoader, SequentialSampler,TensorDataset
torch.manual_seed(3047)

class CGAN_Generator(nn.Module):
    def __init__(self, input_dim, encoder_dim, latent_dim, num_classes, fields):
        # 标签的数量
        super(CGAN_Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim, encoder_dim),
                nn.BatchNorm1d(encoder_dim),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(encoder_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
                nn.Dropout(0.2),


                nn.Linear(latent_dim * 2, encoder_dim),
                nn.BatchNorm1d(encoder_dim),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(encoder_dim, input_dim)
            ]
        )
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.fields = fields

    def forward(self, x, label):
        label = label.long()
        for idx in range(3):
            if idx == 2:
                label = self.label_emb(label).squeeze(1)
                x = torch.cat((x, label), dim=1)
            x = self.linears[4*idx](x)
            x = self.linears[4*idx+1](x)
            x = self.linears[4 * idx + 2](x)
            x = self.linears[4 * idx + 3](x)
        x = self.linears[-1](x)
        current_ind = 0
        decodes = []
        for i in range(len(self.fields)):
            if self.fields[i].data_type == "Categorical Data":
                dim = self.fields[i].dim()
                data = nn.functional.softmax(x[:, current_ind:current_ind + dim], dim=1)
                decodes.append(data)
                current_ind = current_ind + dim
            else:
                decodes.append(self.sigmod(x[:, current_ind:current_ind + 1]))
                current_ind = current_ind + 1
        decodes = torch.cat(decodes, dim=1)
        return decodes


class CGAN_Discriminator(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes, out_dim):
        super(CGAN_Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = torch.nn.Linear(input_dim, latent_dim)
        self.fc2 = torch.nn.Linear(2 * latent_dim, input_dim)
        self.fc3 = torch.nn.Linear(input_dim, out_dim)
        self.batch_normal1 = nn.BatchNorm1d(latent_dim)
        self.batch_normal2 = nn.BatchNorm1d(out_dim)
        self.relu = torch.nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)


    def forward(self, new_x, label):
        label = label.long()
        inp = new_x
        out = self.fc1(inp)
        out = self.dropout(self.relu(out))
        out = torch.cat((out, self.label_emb(label).squeeze(1)), dim=1)
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.sigmoid(self.fc3(out))
        return out

def train_CGAN(discriminator, generator, FDs_model_list, epochs, lr, batch_size, loss_weight, data_m, impute_data_code, label_data, fields, value_cat, values,miss_data_x,enc,ori_data,continuous_cols,label_num,device,use_Learner):
    print("------------------Diffusion Discriminator train--------------------")
    eq_dict = get_eq_dict(values, miss_data_x.copy(), data_m)
    impute_data = None
    torch.manual_seed(3047)
    if device == torch.device('cuda:0') or device == torch.device('cuda:1'):
        generator.to(device)
        discriminator.to(device)
    M_tensor = get_M_by_data_m(data_m, fields, device)
    zero_feed_data = M_tensor * impute_data_code
    x = impute_data_code.to(device)
    # Retrieve the validation set of the learner.
    if use_Learner == 'True' or True:
        valid_data_index = get_valid_data_index(data_m, discriminator, impute_data_code, device)
        train_data_index = [i for i in range(len(data_m)) if i not in valid_data_index]
        valid_data_code = zero_feed_data[valid_data_index]
        label_data_code = torch.FloatTensor(label_data.values).to(device)
        x_valid = x[valid_data_index]
        y_valid = label_data_code[valid_data_index]

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0.0001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)
    m_data = torch.tensor(data_m).float().to(device)
    discriminator.train()
    generator.train()
    no, dim = x.shape
    noise = torch.randn_like(x) * 0.1
    x = 0.9 * x + noise
    indices_tensor = torch.tensor(range(len(x)))
    dataset = TensorDataset(x, label_data_code, indices_tensor)
    dataload = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_epochs = epochs
    L_loss = 0
    cost_time = 0
    rmse_max = 9999
    b_e = 0
    eval_time = 0
    best_time = 0
    impute_data = 0
    # total_epochs = np.round(total_epochs / (max(x.shape[0] / 1000, 1)) / 100)*100
    # total_epochs = 10
    for epoch in tqdm(range(total_epochs), total=total_epochs):
        other_loss = L_loss * loss_weight['L_weight']
        generator.train()
        # 训练generator
        loss_record = MeanMetric()
        FD_loss = 0
        for X_0_batch, label_data, sample_data_index in dataload:
            # 向X_0_batch加噪声
            M_sample = M_tensor[sample_data_index]
            m_sample = m_data[sample_data_index]
            zero_feed_sample = zero_feed_data[sample_data_index]
            generate_out_x = generator(X_0_batch, label_data)
            Res_loss = res_loss_func(fields, generate_out_x, X_0_batch, M_sample)
            decoder_z_impute = zero_feed_sample + (1 - M_sample) * generate_out_x
            discriminator_z = discriminator(decoder_z_impute, label_data)
            G_D_loss = (-torch.mean((1 - m_sample) * torch.log(discriminator_z + 1e-8)))
            FD_loss = Diffusion_FD_loss(decoder_z_impute, FDs_model_list, fields)
            # FD_loss = 0
            loss = Res_loss * loss_weight['Res_weight'] + G_D_loss * loss_weight['G_D_weight'] + FD_loss * loss_weight['FD_weight'] + other_loss
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            generate_out_x = generator(X_0_batch, label_data)
            decoder_z_impute = zero_feed_sample + (1 - M_sample) * generate_out_x
            discriminator_z_pro = discriminator(decoder_z_impute, label_data)
            discriminator_z_loss = -torch.mean(m_sample * torch.log(discriminator_z_pro + 1e-8) + (1 - m_sample) * torch.log(
        1. - discriminator_z_pro + 1e-8))
            discriminator_loss = discriminator_z_loss * loss_weight['VAE_D_weight']  
            optimizer_D.zero_grad()
            discriminator_loss.backward()
            optimizer_D.step()
            
        if epoch % 100 == 0 and epoch > 0:
            generator.eval()
            discriminator.eval()
            generate_x = generator(x, label_data_code)
            code = zero_feed_data + (1 - M_tensor) * generate_x
            rmse, mae = test_impute_data_rmse(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc,
                                              ori_data, continuous_cols)
            if rmse < rmse_max:
                impute_data = reconvert_data(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc)
                rmse_max = rmse
                b_e = epoch
                best_time = time.time()
            if use_Learner == 'True':
                x_train_code = code[train_data_index]
                y_train_code = label_data_code[train_data_index]
                L_loss, acc = train_L_code(1000, x_train_code, y_train_code, x_valid, y_valid, label_num, device)
            else:
                L_loss = 0
    # if impute_data == 0:
    #     generator.eval()
    #     discriminator.eval()
    #     generate_x = generator(x, label_data_code)
    #     code = zero_feed_data + (1 - M_tensor) * generate_x
    #     rmse, mae = test_impute_data_rmse(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc,
    #                                                 ori_data, continuous_cols)
    #     impute_data = reconvert_data(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc)
    print('效果最好的轮次为：{}'.format(b_e))

    # Ensure impute_data is a valid DataFrame
    if impute_data is None or impute_data == 0:
        print("Warning: No valid imputation found during training, using final generated data")
        # Use the final generated code to create imputation
        generator.eval()
        discriminator.eval()
        generate_x = generator(x, label_data_code)
        code = zero_feed_data + (1 - M_tensor) * generate_x
        impute_data = reconvert_data(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc)

    fill_np = impute_data.values
    fill_np = fill_data_mean(fill_np, continuous_cols)
    return fill_np

