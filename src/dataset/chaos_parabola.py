import numpy as np
import torch.utils.data as data

mu = 2


class ChaosParabolaDataSet(data.Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        z = np.random.rand(1, 32, 32) * 2 - 1
        seq = []
        for jx in range(12):
            z = z + np.random.normal(scale=0.01, size=(32, 32))
            z = z * (z > -1.0) * (z < 1.0)
            z = 1 - mu * z * z
            seq.append(z.reshape(1, 32, 32))
        seq = np.concatenate(seq, axis=0)
        return seq[0:2], seq[2:12]

    def __len__(self):
        return self.length

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str
