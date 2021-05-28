import torch
import torch.nn as nn


class SelfAttentionLayer(nn.Module):
    def __init__(self, c_in):
        super(SelfAttentionLayer, self).__init__()
        self.c_in = c_in

        self.qury_conv = nn.Conv1d(c_in, c_in//8, 1)
        self.key_conv = nn.Conv1d(c_in, c_in//8, 1)
        self.value_conv = nn.Conv1d(c_in, c_in, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x.squeeze(3).squeeze(3)
        # batch_size, n_channel, t = x.size()

        proj_qury = self.qury_conv(x).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        energy = torch.bmm(proj_qury, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.gamma*out+x
        return out.unsqueeze(3).unsqueeze(3), attention


class CrossAttentinLayer(nn.Module):
    def __init__(self, c_in):
        super(CrossAttentinLayer, self).__init__()
        self.c_in = c_in

        self.qury_conv = nn.Conv1d(c_in, c_in // 8, 1)
        self.key_conv = nn.Conv1d(c_in, c_in // 8, 1)
        self.value_conv = nn.Conv1d(c_in, c_in, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))   #TODO: limited in [0,1]?

    def forward(self, x, y):
        x = x.squeeze(3).squeeze(3)
        y = y.squeeze(3).squeeze(3)

        proj_qury = self.qury_conv(y).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        energy = torch.bmm(proj_qury, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(y)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.gamma * out + x

        return out.unsqueeze(3).unsqueeze(3), attention


class SelectAttentionLayer(nn.Module):
    def __init__(self, c_in):
        super(SelectAttentionLayer, self).__init__()
        self.conv1 = nn.Conv1d(c_in*2, c_in*4, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(c_in*4)
        self.conv2 = nn.Conv1d(c_in*4, c_in*8, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(c_in*8)
        self.fc1 = nn.Linear(c_in*8, c_in)
        # self.fc1 = nn.Linear(c_in * 2, c_in)
        self.fc2 = nn.Linear(c_in, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.7)

        self.cross_module = CrossAttentinLayer(c_in)

    def forward(self, x, y):
        x = x.squeeze(3).squeeze(3)
        y = y.squeeze(3).squeeze(3)
        b, c, t = x.size()
        comb = torch.cat((x, y), dim=1)
        z = self.conv1(comb)
        # z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)
        # z = self.bn2(z)
        z = torch.max(z, dim=2)[0]
        z = self.fc1(z)
        z = self.drop(z)
        z = self.relu(z)
        z = self.fc2(z)
        w = self.sigmoid(z)

        w = w.repeat(1, c).unsqueeze(2)
        new_x = w*x+(1-w)*y
        new_y = w*y+(1-w)*x

        new_x = new_x.unsqueeze(3).unsqueeze(3)
        new_y = new_y.unsqueeze(3).unsqueeze(3)

        return self.cross_module(new_x, new_y)
