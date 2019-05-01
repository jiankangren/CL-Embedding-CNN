import re
import os
import random
import tarfile
from six.moves import urllib
#from torchtext import data
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import codecs
#import mydatasets
import h5py
import csv
import pandas as pd
import numpy as np

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

class EncodedURLDataset(Dataset):
    def __init__(self, in_file, phase):
        super(EncodedURLDataset, self).__init__()

        self.file = h5py.File(in_file, 'r')[phase]
        self.n_vec = self.file['x'].shape[0]

    def __getitem__(self, index):
        vec = self.file['x'][index]
        label = self.file['y'][index]
        return vec, label

    def __len__(self):
        return self.n_vec

if __name__ == "__main__":

    # a = [1,2,3,4]
    # with open("data/test.txt", "a") as f:
    #     f.write(str(a))
    #     f.write("\t" + str(1) + "\n")
    # data = pd.read_csv("data/URLs.csv", sep="\t")
    # print(data["vec"])

    file = "/media/wislab/Documents/clcnn/char-cnn-text-classification-pytorch-master/req_vocab/CSIC_test_1000_encoded_del.h5py"
    mydataset = EncodedURLDataset(file, "test")



    # mydataset = EncodedURLDataset("/run/media/w/B6C0B7D5C0B799D7/w2v/char-cnn-text-classification-pytorch-master/vocab/360k_encoded.h5py", "train")
    dataloader = DataLoader(mydataset)
    it = iter(dataloader)
    print(it.__len__())



    pass