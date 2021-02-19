import os
import arrow
import logging
import argparse

import numpy as np
import torch
import torch as th
import torch.nn as nn

from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from leibniz.unet.senet import SELayer

from dataset.moving_mnist import MovingMNIST


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=str, default='-1', help="index of gpu")
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


class MMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.iconv = nn.Conv2d(10, 40, kernel_size=5, padding=2)
        self.oconv = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.fconvs = nn.ModuleList()
        self.rconvs = nn.ModuleList()
        self.bnorms = nn.ModuleList()
        self.senets = nn.ModuleList()
        for ix in range(10):
            self.fconvs.append(nn.Conv2d(40, 40, kernel_size=5, padding=2))
            self.bnorms.append(nn.BatchNorm2d(40, affine=True))
            self.senets.append(SELayer(40))
        for ix in range(10):
            self.rconvs.append(nn.Conv2d(40, 20, kernel_size=3, padding=1))

    def forward(self, input):
        input = input / 255.0
        output = th.zeros_like(input)
        flow = self.iconv(input)
        for ix in range(10):
            flow = self.fconvs[ix](flow)
            flow = self.relu(flow)
            flow = self.bnorms[ix](flow)
            flow = self.senets[ix](flow)
            param = self.rconvs[ix](flow)
            output = (output + param[:, 0:10]) * param[:, 10:20] * input

        return self.oconv(output) * 255.0


mdl = MMModel()
mse = nn.MSELoss()
mae = lambda x, y: th.mean(th.abs(x - y), dim=(0, 1)).sum()
optimizer = th.optim.Adam(mdl.parameters())


def train(epoch):
    train_size = 0
    loss_mse = 0.0
    loss_mae = 0.0
    total_ssim = 0.0
    mdl.train()
    for step, sample in enumerate(train_loader):
        logger.info(f'-----------------------------------------------------------------------')
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
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | MSE Loss: {loss.item()}')
        loss_mse += loss.item() * batch
        train_size += batch

        loss = mae(result, target)
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | MAE Loss: {loss.item()}')
        loss_mae += loss.item() * batch

        sim = 0.0
        for ix in range(0, target.shape[0]):
            for jx in range(0, target.shape[1]):
                imgx = result[ix, jx].detach().cpu().numpy()
                imgy = target[ix, jx].detach().cpu().numpy()
                sim += ssim(imgx, imgy) / (imgx.shape[0] * imgx.shape[1])
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | SSIM: {sim}')
        total_ssim += sim

    logger.info(f'======================================================================')
    logger.info(f'Epoch: {epoch + 1:03d} | Train MSE Loss: {loss_mse / train_size}')
    logger.info(f'Epoch: {epoch + 1:03d} | Test MAE Loss: {loss_mae/ train_size}')
    logger.info(f'Epoch: {epoch + 1:03d} | Train SSIM: {total_ssim / train_size}')
    logger.info(f'======================================================================')


def test(epoch):
    mdl.eval()
    test_size = 0
    loss_mse = 0.0
    loss_mae = 0.0
    total_ssim = 0.0
    for step, sample in enumerate(test_loader):
        input, target = sample
        input, target = input.float(), target.float()
        if th.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            mdl.cuda()

        with th.no_grad():
            result = mdl(input)
            batch = result.size()[0]
            test_size += batch

            loss = mse(result, target)
            loss_mse += loss.item() * batch

            loss = mae(result, target)
            loss_mae += loss.item() * batch

            sim = 0.0
            for ix in range(0, target.shape[0]):
                for jx in range(0, target.shape[1]):
                    imgx = result[ix, jx].detach().cpu().numpy()
                    imgy = target[ix, jx].detach().cpu().numpy()
                    sim += ssim(imgx, imgy) / (imgx.shape[0] * imgx.shape[1])
            total_ssim += sim

    logger.info(f'======================================================================')
    logger.info(f'Epoch: {epoch + 1:03d} | Test MSE Loss: {loss_mse / test_size}')
    logger.info(f'Epoch: {epoch + 1:03d} | Test MAE Loss: {loss_mae/ test_size}')
    logger.info(f'Epoch: {epoch + 1:03d} | Test SSIM: {total_ssim / test_size}')
    logger.info(f'======================================================================')


if __name__ == '__main__':
    for epoch in range(opt.n_epochs):
        try:
            train(epoch)
            test(epoch)
        except Exception as e:
            logger.exception(e)
            break
