import os
import arrow
import logging
import argparse

import numpy as np
import torch
import torch as th
import torch.nn as nn
import cv2

from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from leibniz.unet import resunet, SELayer
from leibniz.nn.activation import CappingRelu
from leibniz.unet.hyperbolic import HyperBottleneck

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
model_path = Path(f'./_log3-{time_str}')
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
                shuffle=True)


class MMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.enc = resunet(10, 62, block=HyperBottleneck, relu=nn.ReLU(inplace=True), layers=6, ratio=-2,
                            vblks=[1, 1, 1, 1, 1, 1], hblks=[3, 3, 3, 3, 3, 3],
                            scales=[-1, -1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1, 1],
                            spatial=(64, 64))
        self.dec = resunet(10, 10, block=HyperBottleneck, relu=CappingRelu(), layers=6, ratio=-3,
                            vblks=[1, 1, 1, 1, 1, 1], hblks=[0, 0, 0, 0, 0, 0],
                            scales=[-1, -1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1, 1],
                            spatial=(64, 64), final_normalized=True)

    def forward(self, input):
        input = input / 255.0
        b, c, w, h = input.size()

        enc = self.enc(input)
        uprm, vprm, clz, flow = enc[:, 0:10], enc[:, 10:20], enc[:, 20:22], enc[:, 22:]
        flow = flow.view(-1, 10, 2, 2, 64, 64)

        filter = th.zeros(b, 10, w, h)
        if th.cuda.is_available():
            filter = filter.cuda()

        for ix in range(2):
            aprm = flow[:, :, ix, 0]
            mprm = flow[:, :, ix, 1]
            filter = (filter + aprm * uprm) * (1 + mprm * vprm)

        clazzz = self.relu(clz).view(-1, 1, 2, 64, 64, 1, 1)
        filter = self.relu(self.dec(filter)).view(-1, 10, 1, 1, 1, 64, 64)
        output = th.sum(filter * clazzz, dim=(2, 3, 4)).view(-1, 10, 64, 64)

        return output * 255.0


mdl = nn.DataParallel(MMModel(), output_device=0)
mse = nn.MSELoss()

evl_mse = lambda x, y: th.mean((x - y)**2, dim=(0, 1)).mean()
evl_mae = lambda x, y: th.mean(th.abs(x - y), dim=(0, 1)).mean()
optimizer = th.optim.Adam(mdl.parameters())


def train(epoch):
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

        loss = evl_mse(result, target).detach()
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | MSE Loss: {loss.item()}')
        loss_mse += loss.item()

        loss = evl_mae(result, target).detach()
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | MAE Loss: {loss.item()}')
        loss_mae += loss.item()

        sim = 0.0
        for ix in range(0, target.shape[0]):
            for jx in range(0, target.shape[1]):
                imgx = result[ix, jx].detach().cpu().numpy()
                imgy = target[ix, jx].detach().cpu().numpy()
                sml = ssim(imgx, imgy)
                sim += sml / (target.shape[0] * target.shape[1])
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | SSIM: {sim}')
        total_ssim += sim

        if step == len(train_loader) - 1:
            for ix in range(10):
                img = input[0, ix].detach().cpu().numpy()
                cv2.imwrite('%s/train_i_%02d.png' % (model_path, ix), img)
            for ix in range(10):
                img = target[0, ix].detach().cpu().numpy()
                cv2.imwrite('%s/train_r_%02d.png' % (model_path, ix), img)
            for ix in range(10):
                img = result[0, ix].detach().cpu().numpy()
                cv2.imwrite('%s/train_p_%02d.png' % (model_path, ix), img)

    logger.info(f'======================================================================')
    train_size = len(train_loader)
    logger.info(f'Epoch: {epoch + 1:03d} | Train MSE Loss: {loss_mse / train_size}')
    logger.info(f'Epoch: {epoch + 1:03d} | Test MAE Loss: {loss_mae/ train_size}')
    logger.info(f'Epoch: {epoch + 1:03d} | Train SSIM: {total_ssim / train_size}')
    logger.info(f'======================================================================')


def test(epoch):
    mdl.eval()
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

            loss = evl_mse(result, target)
            loss_mse += loss.item()

            loss = evl_mae(result, target)
            loss_mae += loss.item()

            sim = 0.0
            for ix in range(0, target.shape[0]):
                for jx in range(0, target.shape[1]):
                    imgx = result[ix, jx].detach().cpu().numpy()
                    imgy = target[ix, jx].detach().cpu().numpy()
                    sim += ssim(imgx, imgy) / (target.shape[0] * target.shape[1])
            total_ssim += sim

            if step == len(test_loader) - 1:
                for ix in range(10):
                    img = input[0, ix].detach().cpu().numpy()
                    cv2.imwrite('%s/test_i_%02d.png' % (model_path, ix), img)
                for ix in range(10):
                    img = target[0, ix].detach().cpu().numpy()
                    cv2.imwrite('%s/test_r_%02d.png' % (model_path, ix), img)
                for ix in range(10):
                    img = result[0, ix].detach().cpu().numpy()
                    cv2.imwrite('%s/test_p_%02d.png' % (model_path, ix), img)

    logger.info(f'======================================================================')
    test_size = len(test_loader)
    simval = total_ssim / test_size
    logger.info(f'Epoch: {epoch + 1:03d} | Test MSE Loss: {loss_mse / test_size}')
    logger.info(f'Epoch: {epoch + 1:03d} | Test MAE Loss: {loss_mae/ test_size}')
    logger.info(f'Epoch: {epoch + 1:03d} | Test SSIM: {total_ssim / test_size}')
    logger.info(f'======================================================================')

    th.save(mdl.state_dict(), model_path / f'm_ssim{simval:0.8f}_epoch{epoch + 1:03d}.mdl')
    glb = list(model_path.glob('*.mdl'))
    if len(glb) > 6:
        for p in sorted(glb)[:1]:
            os.unlink(p)


if __name__ == '__main__':
    for epoch in range(opt.n_epochs):
        try:
            train(epoch)
            test(epoch)
        except Exception as e:
            logger.exception(e)
            break
