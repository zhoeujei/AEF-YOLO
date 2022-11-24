import math

from torch import nn
import torch.nn.functional as F

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        std_out_channel = x.view(-1, int(x.shape[1]), int(x.shape[2]) * int(x.shape[2])).std(dim=2).view(-1, int(x.shape[1]), 1, 1)
        y = y * std_out_channel
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return y.expand_as(x)


class se_block_num(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block_num, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, 16, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        std_out_channel = x.view(-1, int(x.shape[1]), int(x.shape[2]) * int(x.shape[2])).std(dim=2).view(-1, int(x.shape[1]))
        y = y * std_out_channel
        y = self.fc(y).view(b, 1, 1, 1)
        return y

class se_block_two_num(nn.Module):
    def __init__(self, channel, ratio=4):
            super(se_block_two_num, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),

                nn.Linear(channel // ratio, (channel // ratio)//ratio, bias=False),
                nn.ReLU(inplace=True),

                nn.Linear((channel // ratio)//ratio, 2, bias=False),

            )

    def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            std_out_channel = x.view(-1, int(x.shape[1]), int(x.shape[2]) * int(x.shape[2])).std(dim=2).view(-1, int(
                x.shape[1]))
            y = y * std_out_channel
            y =  F.softmax(self.fc(y)).view(b, 2, 1, 1)
            return y



