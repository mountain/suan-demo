import torch as th
import torch.nn as nn


class DirectBlocks(nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(DirectBlocks, self).__init__()
        self.dim = dim
        self.step = step

        if relu is None:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = relu

        if conv is None:
            self.conv = nn.Conv2d
        else:
            self.conv = conv

        self.conv0 = self.conv(dim, dim, kernel_size=3, padding=1, bias=False)
        self.conv1 = self.conv(dim, dim, kernel_size=3, padding=1, bias=False)
        self.conv2 = self.conv(dim, dim, kernel_size=3, padding=1, bias=False)
        self.conv3 = self.conv(dim, dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        a = self.conv0(x)
        a = self.relu(a)
        a = self.conv1(a)

        b = self.conv2(x)
        b = self.relu(b)
        b = self.conv3(b)

        return a * x + b
