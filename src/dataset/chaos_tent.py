import numpy as np
import torch.utils.data as data


class ChaosTentDataSet(data.Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, index):
        z = np.random.rand(32, 32)
        seq = [z]
        for jx in range(11):
            z = (z < 0.5) * (2 * z) + (z >= 0.5) * (2 - 2 * z) + np.random.rand(32, 32) * 0.01
            z = z * (z < 1.0) * (z > 0.0)
            seq.apeend(z.reshape(1, 256, 256))
        seq = np.concatenate(seq, axis=0)
        return seq[0:2], seq[2:12]

    def __len__(self):
        return self.length

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str
