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
from torch.autograd import Variable
from leibniz.unet import resunet
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
                shuffle=True)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(10, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU6(inplace=True)
        self.oconv = nn.Conv2d(20, 10, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=0.5)

        self.enc = resunet(10, 160, block=HyperBottleneck, layers=6, ratio=-1,
                vblks=[1, 1, 1, 1, 1, 1], hblks=[3, 3, 3, 3, 3, 3],
                scales=[-1, -1, -1, -1, -1, -1], factors=[1, 1, 1, 1, 1, 1],
                spatial=(64, 64))

        self.dec = lambda x: self.relu6(self.oconv(self.relu(x))) / 6

    def forward(self, input, noise):
        input = input / 255.0 + noise
        b, c, w, h = input.size()
        flow = self.enc(input).view(-1, 20, 2, 4, 64, 64)

        output = th.zeros(b, 20, w, h)
        if th.cuda.is_available():
            output = output.cuda()

        for ix in range(2):
            aparam = flow[:, :, ix, 0]
            mparam = flow[:, :, ix, 1]
            uparam = flow[:, :, ix, 2]
            vparam = flow[:, :, ix, 3]
            output = (output + aparam * uparam) * (1 + mparam * vparam)
            if ix < 2 - 1:
                output = self.dropout(output)

        output = self.dec(output)
        return output * 255.0


generator = nn.DataParallel(Generator(), output_device=0)
discriminator = nn.DataParallel(Discriminator(), output_device=0)
adversarial_loss = torch.nn.BCELoss()
mse = nn.MSELoss()

if th.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    mse.cuda()


evl_mse = lambda x, y: th.mean((x - y)**2, dim=(0, 1)).mean()
evl_mae = lambda x, y: th.mean(th.abs(x - y), dim=(0, 1)).mean()

optimizer_G = torch.optim.Adam(generator.parameters())
optimizer_D = torch.optim.Adam(discriminator.parameters())


Tensor = torch.cuda.FloatTensor if th.cuda.is_available() else torch.FloatTensor


def train(epoch):
    generator.train()
    discriminator.train()
    loss_mse = 0.0
    loss_mae = 0.0
    total_ssim = 0.0
    for step, sample in enumerate(train_loader):
        logger.info(f'-----------------------------------------------------------------------')
        input, target = sample
        input, target = input.float(), target.float()
        if th.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # Adversarial ground truths
        valid = Variable(Tensor(input.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(input.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(input.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (input.shape[0], 10, 64, 64))))

        # Generate a batch of images
        result = generator(input, z)

        # Loss measures generator's ability to fool the discriminator
        loss = mse(result, target) / 25
        a_loss = adversarial_loss(discriminator(result), valid)
        g_loss = loss + a_loss
        logger.info(f'-----------------------------------------------------------------------')
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | pred: {loss.item()}')
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | advs: {a_loss.item()}')
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | genr: {g_loss.item()}')

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(result.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        logger.info(f'-----------------------------------------------------------------------')
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | real: {real_loss.item()}')
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | fake: {fake_loss.item()}')
        logger.info(f'Epoch: {epoch + 1:03d} | Step: {step + 1:03d} | totl: {d_loss.item()}')

        d_loss.backward()
        optimizer_D.step()

        logger.info(f'-----------------------------------------------------------------------')
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
                sim += sml / (imgx.shape[0] * imgx.shape[1])
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
    generator.eval()
    loss_mse = 0.0
    loss_mae = 0.0
    total_ssim = 0.0
    for step, sample in enumerate(test_loader):
        input, target = sample
        input, target = input.float(), target.float()
        if th.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            generator.cuda()

        with th.no_grad():
            result = generator(input, 0)

            loss = evl_mse(result, target)
            loss_mse += loss.item()

            loss = evl_mae(result, target)
            loss_mae += loss.item()

            sim = 0.0
            for ix in range(0, target.shape[0]):
                for jx in range(0, target.shape[1]):
                    imgx = result[ix, jx].detach().cpu().numpy()
                    imgy = target[ix, jx].detach().cpu().numpy()
                    sim += ssim(imgx, imgy) / (imgx.shape[0] * imgx.shape[1])
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

    th.save(generator.state_dict(), model_path / f'm_ssim{simval:0.8f}_epoch{epoch + 1:03d}.mdl')
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
