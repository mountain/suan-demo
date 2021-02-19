from __future__ import print_function
import argparse
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset.yago import Yago2Dataset
from torch.optim.lr_scheduler import StepLR


class Embeding(nn.Module):
    def __init__(self, node_num, link_num, device):
        super(Embeding, self).__init__()
        self.node_num = node_num
        self.link_num = link_num

        self.nodes = nn.Parameter(th.rand((node_num)).to(device))
        self.x = nn.Parameter((2 * th.rand(link_num) - 1).to(device))
        self.y = nn.Parameter((2 * th.rand(link_num) - 1).to(device))

    def forward(self, src, lnk):
        dx = self.x[lnk]
        dy = self.y[lnk]
        val = self.nodes[src]
        return dx + dy * val


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        src, lnk, tgt = data[:, 0], data[:, 1], data[:, 2]
        optimizer.zero_grad()
        output = model(src, lnk)
        loss = F.mse_loss(output, model.nodes[tgt])
        loss.backward(retain_graph=True)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\tLoss: {:.6f}\tx: {:.6f} {:.6f}\ty: {:.6f} {:.6f}'.format(epoch, loss.item(),
                                                         th.min(model.x).detach().item(), th.max(model.x).detach().item(),
                                                         th.min(model.y).detach().item(), th.max(model.y).detach().item(),))
            if args.dry_run:
                break


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Embeding')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and th.cuda.is_available()

    th.manual_seed(args.seed)

    device = th.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    dataset1 = Yago2Dataset('data/yago2.npz')
    train_loader = th.utils.data.DataLoader(dataset1, **train_kwargs, shuffle=True)

    model = Embeding(9656374, 83, device)
    optimizer = optim.Adam(model.parameters())

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

    if args.save_model:
        th.save(model.state_dict(), "yago.pt")


if __name__ == '__main__':
    main()
