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
from leibniz.unet.complex_hyperbolic import CmplxHyperBottleneck
from leibniz.unet.hyperbolic import HyperBottleneck
from leibniz.unet.senet import SEBottleneck
from leibniz.nn.activation import CappingRelu
from leibniz.nn.normalizor import PWLNormalizor

from dataset.chaos_tent import ChaosTentDataSet


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=str, default='0', help="index of gpu")
parser.add_argument("-c", "--n_cpu", type=int, default=64, help="number of cpu threads to use during batch generation")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("-e", "--epoch", type=int, default=0, help="current epoch to start training from")
parser.add_argument("-n", "--n_epochs", type=int, default=500, help="number of epochs of training")
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


train_set = ChaosTentDataSet(length=800)
test_set = ChaosTentDataSet(length=200)

batch_size = opt.batch_size

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)


mean_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)


total_sum = 0.0
total_cnt = 0.0
for step, sample in enumerate(mean_loader):
    input, target = sample
    input, target = input.float(), target.float()
    data = th.cat((input, target), dim=1)
    total_sum += data.sum().item()
    total_cnt += np.prod(data.size())

mean = total_sum / total_cnt
print(mean)


std_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

total_std = 0.0
total_cnt = 0.0
for step, sample in enumerate(mean_loader):
    input, target = sample
    input, target = input.float(), target.float()
    data = th.cat((input, target), dim=1)
    total_std += ((data - mean) * (data - mean)).sum().item()
    total_cnt += np.prod(data.size())

std = total_std / total_cnt
print(std)


class LearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.pwln = PWLNormalizor(1, 128)
        self.unet = UNet(2, 10, normalizor='batch', spatial=(32, 32), layers=5, ratio=0,
                            vblks=[2, 2, 2, 2, 2], hblks=[2, 2, 2, 2, 2],
                            scales=[-1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1],
                            block=CmplxHyperBottleneck, relu=CappingRelu(), final_normalized=True)

    def forward(self, input):
        input = (input - mean) / std
        #input = self.pwln(input.reshape(-1, 1, 32, 32)).reshape(-1, 2, 32, 32)
        output = self.unet(input)
        #output = self.pwln.inverse(output.reshape(-1, 1, 32, 32)).reshape(-1, 10, 32, 32)
        output = output * std + mean
        return output


class PerfectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = th.zeros(1)

    def forward(self, input):
        result = []
        z = input[:, 1:2]
        for ix in range(10):
            z = th.where(z < 0.5, 2 * z, 2 - 2 * z)
            result.append(z)
        return th.cat(result, dim=1)


mdl = LearningModel()
pfc = PerfectModel()

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


def baseline(epoch):
    test_size = 0
    loss_per_epoch = 0.0
    for step, sample in enumerate(test_loader):
        input, target = sample
        input, target = input.float(), target.float()
        if th.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            pfc.cuda()

        with th.no_grad():
            result = pfc(input)
            loss = mse(result, target)

            batch = result.size()[0]
            logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | Loss: {loss.item()}')
            loss_per_epoch += loss.item() * batch
            test_size += batch

    logger.info(f'Epoch: {epoch + 1:03d} | Baseline: {loss_per_epoch / test_size}')


if __name__ == '__main__':
    for epoch in range(opt.n_epochs):
        try:
            train(epoch)
            test(epoch)
            baseline(epoch)
        except Exception as e:
            logger.exception(e)
            break
