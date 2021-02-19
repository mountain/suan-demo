import numpy as np

from torch.utils.data import Dataset


class Yago2Dataset(Dataset):
    def __init__(self, path):
        super(Yago2Dataset, self).__init__()
        self.data = np.load(path)['data']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return np.array(self.data[idx], dtype=np.int64)
