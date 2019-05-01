import torch
import torch.nn as nn
from Embedding.CharEmbedding import CharEmbedding

class CharCNN(nn.Module):

    def __init__(self, num_features, w2v_model_path, trainable=False):
        super(CharCNN, self).__init__()
        n_filter = 1024
        self.num_features = int(num_features)
        self.w2v_model_path = w2v_model_path
        self.embeddings = CharEmbedding(num_features, w2v_model_path=self.w2v_model_path).create_emb_layer(trainable)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(5, self.num_features), stride=1),
            nn.ReLU()
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(3, n_filter), stride=1),
            # nn.BatchNorm2d(n_filter),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(3, n_filter), stride=1),
            # nn.BatchNorm2d(n_filter),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(3, n_filter), stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(3, n_filter), stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(1, n_filter, kernel_size=(3, n_filter), stride=1),
            nn.ReLU()
        )

        self.maxpool6 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.fc1 = nn.Sequential(
            #nn.Linear(6144, 2048), for phish
            nn.Linear(39936, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5)
            #nn.Dropout(p=0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5)
            #nn.Dropout(p=0.3)
        )
        self.fc3 = nn.Linear(2048, 2)
        self.softmax = nn.LogSoftmax()
        # nn.LogSoftmax()

        # self.inference_log_softmax = InferenceBatchLogSoftmax()

    def forward(self, x):
        debug = False
        x = self.embeddings(x)

        x = x.unsqueeze(1)
        # if debug:
        #     print('x.size()', x.size())

        x = self.conv1(x)
        if debug:
            print('x after conv1', x.size())

        x = x.transpose(1, 3)
        if debug:
            print('x after transpose', x.size())

        x = self.maxpool1(x)
        if debug:
            print('x after maxpool1', x.size())

        x = self.conv2(x)
        if debug:
            print('x after conv2', x.size())

        x = x.transpose(1, 3)
        if debug:
            print('x after transpose', x.size())

        x = self.maxpool2(x)
        if debug:
            print('x after maxpool2', x.size())

        x = self.conv3(x)
        if debug:
            print('x after conv3', x.size())

        x = x.transpose(1, 3)
        if debug:
            print('x after transpose', x.size())

        x = self.conv4(x)
        if debug:
            print('x after conv4', x.size())

        x = x.transpose(1, 3)
        if debug:
            print('x after transpose', x.size())

        x = x.contiguous().view(x.size(0), -1)
        if debug:
            print('Collapse x:, ', x.size())

        x = self.fc1(x)
        if debug:
            print('FC1: ', x.size())

        x = self.fc2(x)
        if debug:
            print('FC2: ', x.size())

        x = self.fc3(x)
        if debug:
            print('x: ', x.size())

        x = self.softmax(x)
        # x = self.inference_log_softmax(x)

        return x