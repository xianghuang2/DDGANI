from abc import abstractmethod
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
from model.FD_model import get_my_FD_loss, update_FD_models, get_eq_dict, get_trueProObserve
from utils.util import get_M_by_data_m, test_impute_data_rmse, test_impute_data_acc, get_valid_data_index, sample_x, \
    reconvert_data, get_down_acc, test_fd_data_rmse, test_impute_data_Acc
from model.Learner import train_L_code
torch.manual_seed(3047)


# define TimestepEmbedSequential
class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class EMA():
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model, device):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data.to(device) + self.decay * self.shadow[name].to(device)
                self.shadow[name] = new_average.clone().to(device)

class Diffusion_setting:
    def __init__(self, time_steps, shape, device):
        self.time_steps = time_steps
        self.shape = shape
        self.device = device
        self.initialize()

    def initialize(self):
        self.betas = self.get_betas()
        self.betas[0] = 0
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.one_by_sqrt_alpha_s = 1. / torch.sqrt(self.alphas)
        self.sqrt_beta_s = torch.sqrt(self.betas)
        self.alpha_cumulative = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self):
        scale = 1.
        beta_start = scale * 1e-4
        beta_end = 2 * scale * 1e-2
        return torch.linspace(beta_start, beta_end, self.time_steps, dtype=torch.float32, device=self.device)

class Diffusion(nn.Module):
    def __init__(self, input_dim, encoder_dim, latent_dim, n_steps):
        super(Diffusion, self).__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim, encoder_dim),
                nn.ReLU(),
                nn.Linear(encoder_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, encoder_dim),
                nn.ReLU(),
                nn.Linear(encoder_dim, input_dim)
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, encoder_dim),
                nn.Embedding(n_steps, latent_dim),
                nn.Embedding(n_steps, encoder_dim)
            ]
        )
        self.sigmod = nn.Sigmoid()
        self.time_embedding = nn.Embedding(n_steps, input_dim)

    def forward(self, x, t):
        x = x + self.time_embedding(t)
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x = self.linears[2*idx+1](x)
        x = self.linears[-1](x)

        return x


class Generator_x0(nn.Module):
    def __init__(self, input_dim, encoder_dim, latent_dim, n_steps, fields):
        super(Generator_x0, self).__init__()
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

                nn.Linear(latent_dim, encoder_dim),
                nn.BatchNorm1d(encoder_dim),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(encoder_dim, input_dim)
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, encoder_dim),
                nn.Embedding(n_steps, latent_dim),
                nn.Embedding(n_steps, encoder_dim)
            ]
        )
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.fields = fields
        self.time_embedding = nn.Embedding(n_steps, input_dim)
    def forward(self, x, t):
        x = x + self.time_embedding(t)
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[4*idx](x)
            x = self.linears[4*idx+1](x)
            x = self.linears[4 * idx + 2](x)
            x = self.linears[4 * idx + 3](x)
        x = self.linears[-1](x)
        current_ind = 0
        decodes = []
        out_decodes = []
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


def get_trueProG(decodes, fields):
    truePro = torch.zeros(decodes.shape[0], len(fields))
    cur_index = 0
    for index, field in enumerate(fields):
        if field.data_type == "Categorical Data":
            dim = field.dim()
            data = decodes[:, cur_index:cur_index+dim]
            sigmoid_results = data
            max_values, _ = torch.max(sigmoid_results, dim=1, keepdim=True)
            truePro[:, index] = max_values.squeeze(1)
            cur_index = cur_index + dim
        else:
            cur_index = cur_index + 1
    return truePro.cpu().detach().numpy()
def get_truePro_acc(truePro_vae, truePro_observe, continuse_cols, data_m):

    # true_pro = np.sum(truePro_vae * truePro_observe * data_m)
    #
    # con_cols = truePro_vae[:, continuse_cols]
    # falsePro_vae = 1 - truePro_vae
    # falsePro_vae[:, continuse_cols] = con_cols
    #
    # con_cols = truePro_observe[:, continuse_cols]
    # falsePro_observe = 1 - truePro_observe
    # falsePro_observe[:, continuse_cols] = con_cols
    # false_pro = np.sum(falsePro_vae * falsePro_observe * data_m)
    #
    #
    #
    # sum_true = true_pro + false_pro
    # new_data_m = data_m.copy()
    # new_data_m[:, continuse_cols] = 0
    # zhiXinDu_vae = sum_true / (np.sum(new_data_m)+1e-4)
    G_cols_to_sum = [col for col in range(truePro_observe.shape[1]) if col not in continuse_cols]
    G_sums = np.sum(truePro_observe[:, G_cols_to_sum])

    M_cols_to_sum = [col for col in range(data_m.shape[1]) if col not in continuse_cols]
    M_sums = np.sum(truePro_observe[:, M_cols_to_sum])

    zhiXinDu_vae = G_sums / (M_sums + 1e-4)
    return zhiXinDu_vae
def train_one_epoch_Generator(model, discriminator_noise_x,DS, x, optimizer, loss_scaler,other_loss, epoch, total_epochs, timesteps, device, field,M,m_data,loss_weight,FDs_model_list, fields,zero_feed_data):
    # use MeanMetric to log loss
    loss_record = MeanMetric()
    FD_loss = 0
    model.train()
    if epoch % 100 == 0:
        with tqdm(total=x.shape[0] // 128, dynamic_ncols=True) as tq:
            tq.set_description(f"Epoch: {epoch}/{total_epochs}")
            epoch_num = timesteps
            for epo in range(epoch_num):
                X_0_batch, sample_data_index = sample_x(x, 128)
                M_sample = M[sample_data_index]
                m_sample = m_data[sample_data_index]
                zero_feed_sample = zero_feed_data[sample_data_index]
                tq.update(1)
                # Assign a batch of timesteps to each X0 sample
                batch_timesteps = torch.randint(low=2, high=timesteps+1, size=(X_0_batch.shape[0],), device=device)
                # Diffuse the batch of X0 to their required step of t
                X_t_batch, batch_noise = forward_diffusion(DS, X_0_batch, batch_timesteps)
                with amp.autocast():
                    Pred_x0 = model(X_t_batch, batch_timesteps)
                    Res_loss = res_loss_func(field,Pred_x0,X_0_batch,M_sample)
                    G_D_loss = G_d_loss_func(DS,Pred_x0,X_t_batch,X_0_batch,batch_timesteps, discriminator_noise_x, M_sample, batch_noise, m_sample)
                    decoder_z_impute = zero_feed_sample + (1 - M_sample) * Pred_x0
                    FD_loss = Diffusion_FD_loss(decoder_z_impute, FDs_model_list, fields)
                    # FD_loss = 0
                    loss = Res_loss * loss_weight['Res_weight'] + G_D_loss * loss_weight['G_D_weight'] + FD_loss * loss_weight['FD_weight'] + other_loss
                # optimizer and scaler do the loss bp and update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # loss_scaler.scale(loss).backward()
                # loss_scaler.step(optimizer)
                # loss_scaler.update()
                # log the noise predication loss
                loss_value = loss.detach().item()
                loss_record.update(loss_value)
                # tqdm print loss val
                tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")
                tq.set_postfix_str(s=f"G_FLoss: {G_D_loss * loss_weight['G_D_weight']:.4f}")
            # MeanMetric calculate loss mean
            mean_loss = loss_record.compute().item()
            # tqdm print mean_loss val
            # tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
    else:
        epoch_num = timesteps
        for epo in range(epoch_num):
            X_0_batch, sample_data_index = sample_x(x, 128)
            M_sample = M[sample_data_index]
            m_sample = m_data[sample_data_index]
            zero_feed_sample = zero_feed_data[sample_data_index]
            # Assign a batch of timesteps to each X0 sample
            batch_timesteps = torch.randint(low=2, high=timesteps+1, size=(X_0_batch.shape[0],), device=device)
            # Diffuse the batch of X0 to their required step of t
            X_t_batch, batch_noise = forward_diffusion(DS, X_0_batch, batch_timesteps)
            with amp.autocast():
                Pred_x0 = model(X_t_batch, batch_timesteps)
                Res_loss = res_loss_func(field, Pred_x0, X_0_batch, M_sample)
                G_D_loss = G_d_loss_func(DS, Pred_x0, X_t_batch, X_0_batch, batch_timesteps, discriminator_noise_x, M_sample,
                                         batch_noise, m_sample)
                decoder_z_impute = zero_feed_sample + (1 - M_sample) * Pred_x0
                FD_loss = Diffusion_FD_loss(decoder_z_impute, FDs_model_list, fields)
                # FD_loss = 0
                loss = Res_loss * loss_weight['Res_weight'] + G_D_loss * loss_weight['G_D_weight'] + FD_loss * loss_weight['FD_weight'] + other_loss
            # optimizer and scaler do the loss bp and update
            # optimizer.zero_grad(set_to_none=True)
            # loss_scaler.scale(loss).backward()
            # loss_scaler.step(optimizer)
            # loss_scaler.update()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log the noise predication loss
            loss_value = loss.detach().item()
            loss_record.update(loss_value)
            # tqdm print loss val
        # MeanMetric calculate loss mean
        mean_loss = loss_record.compute().item()
    return mean_loss


def train_one_epoch_discriminator_noise_x(generator_x0, discriminator_noise_x, DS, x, optimizer, loss_scaler, other_loss, epoch, total_epochs, timesteps, device, field, M, m_data, loss_weight):
    loss_record = MeanMetric()
    discriminator_noise_x.train()
    if epoch % 100 == 0:
        with tqdm(total=x.shape[0] // 128, dynamic_ncols=True) as tq:
            tq.set_description(f"Epoch: {epoch}/{total_epochs}")
            epoch_num = timesteps
            for epo in range(epoch_num):
                X_0_batch, sample_data_index = sample_x(x, 128)
                M_sample = M[sample_data_index]
                m_sample = m_data[sample_data_index]
                tq.update(1)
                # Assign a batch of timesteps to each X0 sample
                batch_timesteps = torch.randint(low=2, high=timesteps+1, size=(X_0_batch.shape[0],), device=device)
                # Diffuse the batch of X0 to their required step of t
                X_t_batch, batch_noise = forward_diffusion(DS, X_0_batch, batch_timesteps)
                with amp.autocast():
                    Pred_x0 = generator_x0(X_t_batch, batch_timesteps)
                    G_D_loss = D_loss_func(DS, Pred_x0, X_t_batch, X_0_batch, batch_timesteps, discriminator_noise_x, M_sample,batch_noise, m_sample)
                    loss = G_D_loss * loss_weight['D_weight']
                # optimizer and scaler do the loss bp and update
                optimizer.zero_grad(set_to_none=True)
                loss_scaler.scale(loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
                # log the noise predication loss
                loss_value = loss.detach().item()
                loss_record.update(loss_value)
                # tqdm print loss val
                tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")
            # MeanMetric calculate loss mean
            mean_loss = loss_record.compute().item()
            # tqdm print mean_loss val
            tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
    else:
        epoch_num = timesteps
        for epo in range(epoch_num):
            X_0_batch, sample_data_index = sample_x(x, 128)
            M_sample = M[sample_data_index]
            m_sample = m_data[sample_data_index]
            # Assign a batch of timesteps to each X0 sample
            batch_timesteps = torch.randint(low=2, high=timesteps+1, size=(X_0_batch.shape[0],), device=device)
            # Diffuse the batch of X0 to their required step of t
            X_t_batch, batch_noise = forward_diffusion(DS, X_0_batch, batch_timesteps)
            with amp.autocast():
                Pred_x0 = generator_x0(X_t_batch, batch_timesteps)
                G_D_loss = D_loss_func(DS, Pred_x0, X_t_batch, X_0_batch, batch_timesteps, discriminator_noise_x, M_sample,
                                       batch_noise, m_sample)
                loss = G_D_loss
            # optimizer and scaler do the loss bp and update
            optimizer.zero_grad(set_to_none=True)
            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            # log the noise predication loss
            loss_value = loss.detach().item()
            loss_record.update(loss_value)
            # tqdm print loss val
        # MeanMetric calculate loss mean
        mean_loss = loss_record.compute().item()
    return mean_loss


def get(element: torch.Tensor, idxs: torch.Tensor):
    """
    Get values from "element" by index positions (idxs) and
    reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, idxs-1)  # size: B (same as idxs)
    return ele.reshape(-1, 1)  # size: B,1


def forward_diffusion(DS: Diffusion_setting, x_0: torch.Tensor, timestep: torch.Tensor):
    eps = torch.randn_like(x_0)
    mean = get(DS.sqrt_alpha_cumulative, idxs=timestep) * x_0    #[B, 1] * [B, D] = [B, D]
    std_dev = get(DS.sqrt_one_minus_alpha_cumulative, idxs=timestep)
    sample = mean + std_dev * eps
    return sample, eps

def get_xt(DS, x_0, timestep):
    eps = torch.randn_like(x_0)
    mean = get(DS.sqrt_alpha_cumulative, idxs=timestep) * x_0    #[B, 1] * [B, D] = [B, D]
    std_dev = get(DS.sqrt_one_minus_alpha_cumulative, idxs=timestep)
    sample = mean + std_dev * eps
    return sample

def G_D_onestep(generator_x0, DS, x_t, timestep, M, x):
    batch_noise = torch.randn_like(x)
    Pred_x0 = generator_x0(x_t, timestep)
    xtj1_gen_mean = (get(DS.sqrt_alpha_cumulative, idxs=timestep - 1) * get(DS.betas, idxs=timestep) * Pred_x0 + get(DS.sqrt_alphas, idxs=timestep) * (1 - get(DS.alpha_cumulative, idxs=timestep-1)) * x_t) / (1 - get(DS.alpha_cumulative,idxs=timestep))
    xtj1_gen_std = (1 - get(DS.alpha_cumulative,idxs=timestep-1)) * get(DS.betas,idxs=timestep) * batch_noise / (1 - get(DS.alpha_cumulative,idxs=timestep))
    xtj1_g = xtj1_gen_mean + xtj1_gen_std

    std_dev = get(DS.sqrt_one_minus_alpha_cumulative, idxs=timestep - 1)
    x_0_batch_mean = get(DS.sqrt_alpha_cumulative, idxs=timestep - 1) * x
    xtj1_o = x_0_batch_mean + std_dev * batch_noise

    xtj1 = M * xtj1_o + (1 - M) * xtj1_g
    return xtj1

def reverse_diffusion_G_D_sample(generator_x,discriminator_noise_x, DS, timesteps, no, dim, device, x,M_tensor):
    timesteps_cur = timesteps
    x_t = get_xt(DS, x, torch.randint(low=timesteps_cur, high=timesteps_cur+1, size=(no,), device=device))  # the first X_t
    seq = [x_t]
    for time_step in tqdm(iterable=reversed(range(2, timesteps_cur+1)), total=timesteps-1, dynamic_ncols=False,
                          desc="Sampling :: ", position=0):
        timesteps_batch = torch.ones(no, dtype=torch.long, device=device) * time_step
        if time_step > 2:
            x_t = G_D_onestep(generator_x, DS, x_t,timesteps_batch, M_tensor, x)
            seq.append(x_t)
        else:
            x_t = generator_x(x_t,timesteps_batch)
            X_t_batch, batch_noise = forward_diffusion(DS, x, timesteps_batch)
            std_dev = (1 - get(DS.alpha_cumulative,idxs=timesteps_batch-1)) * get(DS.betas,idxs=timesteps_batch) * batch_noise / (1 - get(DS.alpha_cumulative,idxs=timesteps_batch))
            x_0_batch_mean = (get(DS.sqrt_alpha_cumulative, idxs=timesteps_batch - 1) * get(DS.betas, idxs=timesteps_batch) * x_t + get(DS.sqrt_alphas, idxs=timesteps_batch) * (1 - get(DS.alpha_cumulative, idxs=timesteps_batch-1)) * X_t_batch) / (1 - get(DS.alpha_cumulative,idxs=timesteps_batch))
            xtj1 = x_0_batch_mean + std_dev * batch_noise
            discriminator_noise_x.cpu()
            discriminator_out = discriminator_noise_x(xtj1.cpu(), X_t_batch.cpu(), timesteps_batch.cpu())
            discriminator_out = discriminator_out.to(device)
            discriminator_noise_x.to(device)
    return x_t, discriminator_out

def res_loss_func(fields, reconstruct, x, M):
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
            loss_fn = nn.MSELoss()
            mse = loss_fn(pre, label)
            MSE += mse
            MSE_num = MSE_num + 1
        else:
            m = M[:, curr:curr + dim].cpu().detach().numpy()
            good_rows = np.where(np.all(m == 1, axis=1))[0]
            pre = reconstruct[good_rows,  curr:curr + dim] + 1e-6
            lab = x[good_rows, curr:curr + dim]
            label = torch.argmax(lab, dim=1)
            loss = -torch.mean(torch.log(torch.sum(pre * lab, dim=1)))
            BCE += loss
            BCE_num = BCE_num + 1
        curr += dim
    if BCE_num == 0:
        BCE_num = BCE_num + 1
    if MSE_num == 0:
        MSE_num = MSE_num + 1
    return MSE/MSE_num + BCE/BCE_num

def G_d_loss_func(DS,Pred_x0,X_t_batch,X_0_batch, timestep, discriminator_noise_x, M, batch_noise, m_data):
    std_dev = get(DS.sqrt_one_minus_alpha_cumulative, idxs=timestep - 1)
    x_0_batch_mean = get(DS.sqrt_alpha_cumulative, idxs=timestep-1) * X_0_batch

    xtj1_o = x_0_batch_mean + std_dev * batch_noise
    x_0_gen_mean = get(DS.sqrt_alpha_cumulative, idxs=timestep - 1) * Pred_x0
    xtj1_g = x_0_gen_mean + std_dev * batch_noise

    # ele = element.gather(-1, idxs-1)  # size: B (same as idxs)
    # return ele.reshape(-1, 1)  # size: B,1

    xtj1_gen_mean = (get(DS.sqrt_alpha_cumulative, idxs=timestep - 1) * get(DS.betas, idxs=timestep) * Pred_x0 + get(DS.sqrt_alphas, idxs=timestep) * (1 - get(DS.alpha_cumulative, idxs=timestep-1)) * X_t_batch) / (1 - get(DS.alpha_cumulative,idxs=timestep))
    xtj1_gen_std = (1 - get(DS.alpha_cumulative,idxs=timestep-1)) * get(DS.betas,idxs=timestep) * batch_noise / (1 - get(DS.alpha_cumulative,idxs=timestep))
    xtj1_g = xtj1_gen_mean + xtj1_gen_std
    # x = M * X_0_batch + (1 - M) * Pred_x0
    # mean = get(DS.sqrt_alpha_cumulative, idxs=timestep-1) * x  # [B, 1] * [B, D] = [B, D]
    xtj1 = M * xtj1_o + (1 - M) * xtj1_g
    discriminator_out = discriminator_noise_x(xtj1,X_t_batch,timestep)
    loss = -torch.mean((1 - m_data) * torch.log(discriminator_out + 1e-8))
    return loss

def D_loss_func(DS, Pred_x0, X_t_batch, X_0_batch, timestep, discriminator_noise_x, M,batch_noise, m_data):
    std_dev = get(DS.sqrt_one_minus_alpha_cumulative, idxs=timestep - 1)
    x_0_batch_mean = get(DS.sqrt_alpha_cumulative, idxs=timestep-1) * X_0_batch
    xtj1_o = x_0_batch_mean + std_dev * batch_noise
    x_0_gen_mean = get(DS.sqrt_alpha_cumulative, idxs=timestep - 1) * Pred_x0
    xtj1_g = x_0_gen_mean + std_dev * batch_noise
    xtj1_gen_mean = (get(DS.sqrt_alpha_cumulative, idxs=timestep - 1) * get(DS.betas, idxs=timestep) * Pred_x0 + get(DS.sqrt_alphas, idxs=timestep) * (1 - get(DS.alpha_cumulative, idxs=timestep-1)) * X_t_batch) / (1 - get(DS.alpha_cumulative,idxs=timestep))
    xtj1_gen_std = (1 - get(DS.alpha_cumulative,idxs=timestep-1)) * get(DS.betas,idxs=timestep) * batch_noise / (1 - get(DS.alpha_cumulative,idxs=timestep))
    xtj1_g = xtj1_gen_mean + xtj1_gen_std

    x = M * X_0_batch + (1 - M) * Pred_x0
    mean = get(DS.sqrt_alpha_cumulative, idxs=timestep-1) * x  # [B, 1] * [B, D] = [B, D]
    xtj1 = M * xtj1_o + (1 - M) * xtj1_g

    discriminator_out = discriminator_noise_x(xtj1,X_t_batch,timestep)
    a = discriminator_out.cpu().detach().numpy()
    loss = -torch.mean((1 - m_data) * torch.log(1 - discriminator_out + 1e-4) + m_data * torch.log(discriminator_out + 1e-4))
    if loss is None:
        print(1)
    return loss

def Diffusion_FD_loss(Pred_x0, FDs_model_list, fields):
    FD_loss = get_my_FD_loss(FDs_model_list, Pred_x0, fields)
    return FD_loss

def get_cell_true_diffusion(generator_data, zero_feed_data, fields, data_m, continuous_cols,discriminator_out, M_tensor, value_cat, values, miss_data_x, enc, device):

    truePro_gen = get_trueProG(generator_data, fields)
    truePro_observe = get_trueProObserve(generator_data, zero_feed_data, fields, data_m, device)
    truePro_acc_gen = get_truePro_acc(truePro_gen, truePro_observe, continuous_cols, data_m)

    truePro_dis = discriminator_out.cpu().detach().numpy()
    truePro_acc_dis = get_truePro_acc(truePro_dis, truePro_observe, continuous_cols, data_m)
    decoder_z_impute = zero_feed_data + generator_data * (1 - M_tensor)

    vae_cell_acc = truePro_acc_gen * get_trueProG(decoder_z_impute, fields)
    truePro_dis[:, continuous_cols] = 0
    dis_cell_acc = truePro_acc_dis * truePro_dis
    cell_acc = (vae_cell_acc + dis_cell_acc) / (truePro_acc_dis + truePro_acc_gen)
    cell_acc[np.where(data_m == 1)] = 1
    cell_acc[:, continuous_cols] = truePro_dis[:, continuous_cols]
    return cell_acc


def train_diffusion_discriminator(discriminator, generator_x,discriminator_noise_x, FDs_model_list, num_steps,epochs, lr, batch_size, loss_weight, data_m, impute_data_code, label_data, fields, value_cat, values,miss_data_x,enc,ori_data,continuous_cols,label_num,device,use_Learner):
    print("------------------Diffusion Discriminator train--------------------")
    eq_dict = get_eq_dict(values, miss_data_x.copy(), data_m)
    torch.manual_seed(3047)
    if device == torch.device('cuda:0'):
        generator_x.cuda()
        discriminator_noise_x.cuda()
        discriminator.cuda()
    M_tensor = get_M_by_data_m(data_m, fields, device)
    zero_feed_data = M_tensor * impute_data_code
    x = impute_data_code.to(device)
    # Retrieve the validation set of the learner.
    if use_Learner == 'True':
        valid_data_index = get_valid_data_index(data_m, discriminator, impute_data_code, device)
        train_data_index = [i for i in range(len(data_m)) if i not in valid_data_index]
        valid_data_code = zero_feed_data[valid_data_index]
        label_data_code = torch.FloatTensor(label_data.values).to(device)
        x_valid = x[valid_data_index]
        y_valid = label_data_code[valid_data_index]
    optimizer_G = optim.Adam(generator_x.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0.0001)
    optimizer_D = optim.Adam(discriminator_noise_x.parameters(), lr=0.0001)
    m_data = torch.tensor(data_m).float().to(device)
    discriminator.train()
    generator_x.train()
    discriminator_noise_x.train()
    discriminator_loss_list = []
    learner_acc_list = []
    generator_loss_list = []
    no, dim = x.shape
    dataload = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)
    DS = Diffusion_setting(num_steps, dim, device)
    total_epochs = epochs
    L_loss = 0
    cost_time = 0
    rmse_list,mae_list,acc_list = [], [], []
    for epoch in range(1, total_epochs + 1):
        other_loss = L_loss * loss_weight['L_weight']
        generator_x.train()
        train_one_epoch_Generator(generator_x, discriminator_noise_x, DS, x, optimizer_G, GradScaler(),other_loss, epoch, total_epochs, num_steps, device, fields,M_tensor,m_data,loss_weight,FDs_model_list, fields,zero_feed_data)
        train_one_epoch_discriminator_noise_x(generator_x, discriminator_noise_x, DS, x, optimizer_D, GradScaler(), other_loss, epoch, total_epochs, num_steps, device, fields, M_tensor, m_data, loss_weight)
        if epoch % 10 == 0:
            generator_data, discriminator_out = reverse_diffusion_G_D_sample(generator_x, discriminator_noise_x, DS,
                                                                             num_steps, no, dim, device, x, M_tensor)
            code = zero_feed_data + (1 - M_tensor) * generator_data
            if len(FDs_model_list) != 0:
                cell_acc = get_cell_true_diffusion(generator_data, zero_feed_data, fields, data_m, continuous_cols,
                                         discriminator_out, M_tensor, value_cat, values, miss_data_x.copy(), enc, device)
                new_FDs_model_list = update_FD_models(generator_data, zero_feed_data, fields, data_m, M_tensor, value_cat,
                                                      values, miss_data_x.copy(), enc, device, eq_dict, FDs_model_list,
                                                      cell_acc, continuous_cols, cost_time)
                FDs_model_list = new_FDs_model_list
            if len(value_cat)>0:
                current_ind = 0
                for i in range(len(fields)):
                    if fields[i].data_type == "Categorical Data":
                        dim = fields[i].dim()
                        data = nn.functional.softmax(code[:, current_ind:current_ind + dim],dim=1)
                        code[:, current_ind:current_ind + dim] = data
                        current_ind = current_ind + dim
                    else:
                        current_ind = current_ind + 1
            cur_rmse, cur_mae = test_impute_data_rmse(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc,
                                                      ori_data, continuous_cols)
            rmse_list.append(cur_rmse)
            mae_list.append(cur_mae)
            print("Use_observe_u: ARMSE为:{},  AMAE为:{}".format(cur_rmse, cur_mae))
            if use_Learner == 'True' and epoch % 100 == 0:
                x_train_code = code[train_data_index]
                y_train_code = label_data_code[train_data_index]
                L_loss, acc = train_L_code(1000, x_train_code, y_train_code, x_valid, y_valid, label_num, device)
            else:
                acc = 0
                L_loss = 0
    ARMSE,AMAE = min(rmse_list),min(mae_list)

    # test
    #     generator_data, discriminator_out = reverse_diffusion_G_D_sample(generator_x, discriminator_noise_x, DS, num_steps,
    #                                                                      no, dim, device, x, M_tensor)
    #     code = zero_feed_data + (1 - M_tensor) * generator_data
    #     if len(value_cat) > 0:
    #         current_ind = 0
    #         for i in range(len(fields)):
    #             if fields[i].data_type == "Categorical Data":
    #                 dim = fields[i].dim()
    #                 data = nn.functional.softmax(code[:, current_ind:current_ind + dim], dim=1)
    #                 code[:, current_ind:current_ind + dim] = data
    #                 current_ind = current_ind + dim
    #             else:
    #                 current_ind = current_ind + 1
    #     code = zero_feed_data + (1 - M_tensor) * generator_data
    # ARMSE, AMAE = test_impute_data_rmse(code, fields, value_cat, values, miss_data_x.copy(), data_m, enc,
    #                                               ori_data, continuous_cols)
    # return ARMSE,AMAE,Acc
    return ARMSE, AMAE
