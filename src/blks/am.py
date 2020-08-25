import numpy as np
import torch as th
import torch.nn as nn

from leibniz.unet.senet import SELayer


class AMBlocks(nn.Module):
    extension = 1
    least_required_dim = 1

    def __init__(self, dim, step, relu, conv, reduction=16):
        super(AMBlocks, self).__init__()
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

        self.se_theta = SELayer(dim)
        self.se_u = SELayer(dim)

    def forward(self, x):

        theta = self.conv0(x)
        theta = self.relu(theta)
        theta = self.conv1(theta)
        theta = 2 * np.pi * self.se_theta(theta)

        u = self.conv2(x)
        u = self.relu(u)
        u = self.conv3(u)
        u = self.se_u(u)

        return (x + u * th.cos(theta)) * (1 + u * th.sin(theta))