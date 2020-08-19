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
from leibniz.unet.hyperbolic import HyperBottleneck

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
model_path = Path(f'./{name}-{time_str}')
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

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)


class MMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(6, 1, normalizor='batch', spatial=(32, 64 + 2), layers=5, ratio=1,
                            vblks=[4, 4, 4, 4, 4], hblks=[0, 0, 0, 0, 0],
                            scales=[-1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1],
                            block=HyperBottleneck, relu=nn.ReLU(), final_normalized=False)

    def forward(self, input):
        return self.unet(input)


lr = 1e-3
wd = 1e-2
mdl = MMModel()
mse = nn.MSELoss()
optimizer = th.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)


def train(epoch):
    train_size = 0
    loss_per_epoch = 0.0
    mdl.train()
    for step, sample in enumerate(test_loader):
        input, target = sample
        print('Input:  ', input.shape)
        print('Target: ', target.shape)
        if th.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        result = mdl(input)
        loss = mse(result, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch = list(result.values())[0].size()[0]
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')
        loss_per_epoch += loss.item() * batch
        train_size += batch

    logger.info(f'Epoch: {epoch + 1:03d} | Train Loss: {loss_per_epoch / train_size}')


def test(epoch):
    mdl.eval()
    test_size = 0
    loss_per_epoch = 0.0
    inputs, targets, results = None, None, None
    for step, sample in enumerate(test_loader):
        input, target = sample
        if th.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        with th.no_grad():
            result = mdl(input)
            loss = mse(result, target)
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')
            loss_per_epoch += loss.item() * list(result.values())[0].size()[0]

            batch = list(result.values())[0].size()[0]
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')
            loss_per_epoch += loss.item() * batch
            test_size += batch

    logger.info(f'Epoch: {epoch + 1:03d} | Test Loss: {loss_per_epoch / test_size}')


if __name__ == '__main__':
    for epoch in range(100):
        try:
            train(epoch)
            test(epoch)
        except Exception as e:
            logger.exception(e)
            break
