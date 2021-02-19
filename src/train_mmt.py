import os
import arrow
import logging
import argparse

import numpy as np
import torch
import torch as th
import torch.nn as nn

from pathlib import Path
from leibniz.unet.base import UNet
from leibniz.unet.warpped_hyperbolic2d import HyperBottleneck
from leibniz.nn.activation import CappingRelu

from blks.direct import DirectBlocks
from blks.am import AMBlocks

from dataset.moving_mnist import MovingMNIST


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=str, default='0', help="index of gpu")
parser.add_argument("-c", "--n_cpu", type=int, default=64, help="number of cpu threads to use during batch generation")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("-e", "--epoch", type=int, default=0, help="current epoch to start training from")
parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-m", "--model", type=str, default='', help="metrological model to load")
parser.add_argument("-k", "--check", type=str, default='', help="checkpoint file to load")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

print('cudnn:', th.backends.cudnn.version())

np.core.arrayprint._line_width = 150
np.set_printoptions(linewidth=np.inf)

name = opt.model

time_str = arrow.now().format('YYYYMMDD_HHmmss')
model_path = Path(f'./_log-{time_str}')
model_path.mkdir(exist_ok=True)
log_file = model_path / Path('train.log')

logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(str(opt))

root = './data'
if not os.path.exists(root):
    os.mkdir(root)


train_set = MovingMNIST(root='.data/mnist', train=True, download=True)
test_set = MovingMNIST(root='.data/mnist', train=False, download=True)

batch_size = opt.batch_size

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)


class MMTModel(nn.Module):
    def __init__(self, steps, width, height, device):
        super().__init__()
        # batch, value, class, feature, spatio, steps
        self.steps = steps
        self.class_num = 9
        self.spatio_size = width * height
        self.tube_size = 10
        self.feature_size = width * height

        mu = torch.ones((1, 1, 1, 1, self.feature_size, 1), requires_grad=False) / np.e
        lm = torch.linspace(1 / np.e / np.e, 1 + 1 / np.e, steps=self.feature_size, requires_grad=False).resize(1, 1, 1, 1,
                                                                                                                self.feature_size,
                                                                                                                1)
        pathes = torch.ones((1, 1, self.tube_size, 2, 1, steps))

        self.mu = mu.to(device)
        self.lm = lm.to(device)
        self.pathes = nn.Parameter(pathes.to(device))
        self.inits = torch.cat((
            torch.zeros((1, 1, self.tube_size, 1, 1, 1), requires_grad=False),
            torch.ones((1, 1, self.tube_size, 1, 1, 1), requires_grad=False),
        ), dim=1).to(device)

        val = self.inits
        for k in range(steps):
            val = self.step(k, val)
        self.val = val.to(device)

        self.conv0 = nn.Conv2d(10, 1, 3, 1).to(device)

    def step(self, k, val):
        return self.mu * self.lm * self.pathes[:, :, :, 0:1, :, k:k + 1] + self.lm * self.pathes[:, :, :, 1:2, :,
                                                                                     k:k + 1] * val

    def forward(self, img):
        b, _, _, _ = img.size()
        idx = (1 * (img > 0) + 0 * (img < 0)).view(b, 1, 1, 1, self.feature_size, 1)
        expaned_idx = idx.expand(b, 1, self.tube_size, 1, self.feature_size, 1)
        expaned_val = self.val.expand(b, 2, self.tube_size, 1, self.feature_size, 1)
        data = torch.gather(expaned_val, 1, expaned_idx).view(b, self.tube_size, self.feature_size)

        return output.resize(b, self.class_num)


mdl = MMTModel()
mse = nn.MSELoss()
optimizer = th.optim.Adam(mdl.parameters())


def train(epoch):
    train_size = 0
    loss_per_epoch = 0.0
    mdl.train()
    for step, sample in enumerate(train_loader):
        input, target = sample
        input, target = input.float(), target.float()
        if th.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            mdl.cuda()

        result = mdl(input)
        loss = mse(result, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch = result.size()[0]
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')
        loss_per_epoch += loss.item() * batch
        train_size += batch

    logger.info(f'Epoch: {epoch + 1:03d} | Train Loss: {loss_per_epoch / train_size}')


def test(epoch):
    mdl.eval()
    test_size = 0
    loss_per_epoch = 0.0
    for step, sample in enumerate(test_loader):
        input, target = sample
        input, target = input.float(), target.float()
        if th.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            mdl.cuda()

        with th.no_grad():
            result = mdl(input)
            loss = mse(result, target)

            batch = result.size()[0]
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')
            loss_per_epoch += loss.item() * batch
            test_size += batch

    logger.info(f'Epoch: {epoch + 1:03d} | Test Loss: {loss_per_epoch / test_size}')


if __name__ == '__main__':
    for epoch in range(opt.n_epochs):
        try:
            train(epoch)
            test(epoch)
        except Exception as e:
            logger.exception(e)
            break
