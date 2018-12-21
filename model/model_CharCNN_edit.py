import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CharCNN(nn.Module):

    def __init__(self, dim):
        super(CharCNN, self).__init__()
        n_filter = 1024
        self.dim = dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(3, self.dim), stride=1),
            nn.ReLU()
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(3, n_filter), stride=1),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(3, n_filter), stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(3, n_filter), stride=1),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(6144, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc3 = nn.Linear(1024, 2)

        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.transpose(1, 3)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = x.transpose(1, 3)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = x.transpose(1, 3)
        x = self.conv4(x)
        x = x.transpose(1, 3)

        x = x.contiguous().view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

