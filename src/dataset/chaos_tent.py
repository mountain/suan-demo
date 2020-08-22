import torch as th

from PIL import Image
import torch.utils.data as data

import os
import os.path


class ChaosTentDataSet(data.Dataset):
    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set

        if self.train:
            self.train_data = th.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data = th.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        def _transform_time(data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = self.transform(img) if new_data is None else th.cat([self.transform(img), new_data], dim=0)
            return new_data

        if self.train:
            seq, target = self.train_data[index, :2], self.train_data[index, 10:]
        else:
            seq, target = self.test_data[index, :2], self.test_data[index, 10:]

        if self.transform is not None:
            seq = _transform_time(seq)
        if self.target_transform is not None:
            target = _transform_time(target)

        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
