import re
import os
import random
import tarfile
from six.moves import urllib
#from torchtext import data
from torch.utils.data import Dataset, DataLoader, random_split

import h5py


class H5Dataset(Dataset):
    def __init__(self, in_file):
        super(H5Dataset, self).__init__()

        self.file = h5py.File(in_file, 'r')
        self.n_vec = self.file['y_train'].shape[0]

    def __getitem__(self, index):
        vec = self.file['x_train'][index]
        label = self.file['y_train'][index] + 1
        return vec, label

    def __len__(self):
        return self.n_vec


