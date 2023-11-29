
import torch
import torch.nn as nn
import torch.optim as optim





class L(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(L, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim)  # Data + Hint as inputs
        self.fc2 = torch.nn.Linear(input_dim, output_dim)
        self.fc3 = torch.nn.Linear(input_dim//3, output_dim)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.normal = nn.BatchNorm1d(input_dim)
        self.output_dim = output_dim
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)


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
        # inp = m * x + (1 - m) * g
        out = self.fc2(new_x)
        out = self.softmax(out)
        return out

def train_L_code(epoch, x_train, y_train, x_valid, y_valid, label_num, device):
    torch.manual_seed(3407)
    net = L(x_train.size()[1], label_num).to(device)
    optimizer_L = optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epo in range(epoch):
        if epo != epoch-1:
            outputs = net(x_train.detach())
        else:
            outputs = net(x_train)
        loss = criterion(outputs, y_train.squeeze().long())

        if epo != epoch-1:
            optimizer_L.zero_grad()
            loss.backward()
            optimizer_L.step()
        else:
            optimizer_L.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_L.step()
    with torch.no_grad():
        Y = net(x_valid)
        val_loss = criterion(Y, y_valid.squeeze().long())
        _, y_pred = torch.max(Y.data, dim=1)
        accuracy = (y_pred == y_valid.squeeze().long()).sum().item() / y_valid.size(0)
    return val_loss, accuracy




