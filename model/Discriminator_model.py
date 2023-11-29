import torch
import torch.nn as nn


class D(nn.Module):
    def __init__(self, input_dim, latent_dim, out_dim):
        super(D, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.fc2 = torch.nn.Linear(input_dim, input_dim)
        self.fc3 = torch.nn.Linear(input_dim, out_dim)
        self.batch_normal1 = nn.BatchNorm1d(latent_dim)
        self.batch_normal2 = nn.BatchNorm1d(out_dim)
        self.relu = torch.nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)


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

    def forward(self, new_x):
        inp = new_x
        out = self.fc1(inp)
        out = self.dropout(self.relu(out))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.sigmoid(self.fc3(out))
        return out

def init_D(input_dim, latent_dim, d_input_dim, device):
    discriminator = D(input_dim, latent_dim, d_input_dim, device)
    discriminator.initialize_weights()
    return discriminator

class D_z(nn.Module):
    def __init__(self, latent_dim, device):
        super(D_z, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim, int(latent_dim / 2))
        self.fc2 = torch.nn.Linear(int(latent_dim / 2), int(latent_dim / 3))
        self.fc3 = torch.nn.Linear(int(latent_dim / 3), 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.device = device

    def forward(self, new_x):
        inp = new_x
        out = self.fc1(inp)
        out = self.relu(out)
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out

class Discriminator_x_code(nn.Module):
    def __init__(self, input_dim, device):
        super(Discriminator_x_code, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, int(input_dim / 2))
        self.fc2 = torch.nn.Linear(int(input_dim / 2), int(input_dim / 3))
        self.fc3 = torch.nn.Linear(int(input_dim / 3), 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.device = device

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out


class Discriminator_noise_x(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, n_steps):
        super(Discriminator_noise_x, self).__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, output_dim)
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, latent_dim),
                nn.Embedding(n_steps, latent_dim),
                nn.Embedding(n_steps, latent_dim)
            ]
        )
        self.sigmoid = torch.nn.Sigmoid()
    # [10000,2]-->[10000, 128]+t[10000, 128]--->Relu------>[10000,2]
    def forward(self, xtj1, xt, t):
        x = torch.cat((xtj1, xt),dim=1)
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x = x + t_embedding
            # x = torch.cat((x, t_embedding), dim=1)
            x = self.linears[2*idx+1](x)
        x = self.linears[-1](x)
        x = self.sigmoid(x)
        return x



