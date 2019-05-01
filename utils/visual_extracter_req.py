# -- coding: UTF-8 --

import h5py
from Embedding.CharEmbedding_req import CharEmbedding
import torchvision
from matplotlib.ticker import NullFormatter
from torch import nn
import torch
import numpy as np
from torch.utils.data import DataLoader

from mydatasets import H5Dataset, EncodedURLDataset
from torch.autograd import Variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
        # for visualiation
        per_out = []
        x = self.embeddings(x)
        x = x.unsqueeze(1)
        if debug:
            print('x.size()', x.size())
        per_out.append(x)  # embedding
        x = self.conv1(x)
        if debug:
            print('x after conv1', x.size())

        per_out.append(x)  # conv1

        x = x.transpose(1, 3)
        if debug:
            print('x after transpose', x.size())


        x = self.maxpool1(x)
        if debug:
            print('x after maxpool1', x.size())

        #per_out.append(x) # maxpooling 1

        x = self.conv2(x)
        if debug:
            print('x after conv2', x.size())

        per_out.append(x) # conv2

        x = x.transpose(1, 3)
        if debug:
            print('x after transpose', x.size())

        x = self.maxpool2(x)
        if debug:
            print('x after maxpool2', x.size())

        x = self.conv3(x)
        if debug:
            print('x after conv3', x.size())
        per_out.append(x)

        x = x.transpose(1, 3)
        if debug:
            print('x after transpose', x.size())

        x = self.conv4(x)
        if debug:
            print('x after conv4', x.size())
        #per_out.append(x)


        x = x.transpose(1, 3)
        if debug:
            print('x after transpose', x.size())

        x = x.contiguous().view(x.size(0), -1)

        per_out.append(x) # flatten

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
        #x = self.inference_log_softmax(x)

        return x, per_out


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        self.submodule = submodule

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]

def get_data_new(filename, phase):
    file = h5py.File(filename, "r")
    good = file[phase]["good"][1:]
    bad = file[phase]["bad"][1:]
    all = np.vstack((good, bad))
    file.close()
    return all


def get_data(filename, phase, label):
    file = h5py.File(filename, "r")
    x = file[phase][label][:2000]
    num = file[phase][label].shape[0]
    dim = file[phase][label].shape[1]
    file.close()
    return x, num, dim


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    #return fig

def draw(feature_name, layer):
    l_x, l_y = t_sne(feature_name, layer, "good", 0)
    p_x, p_y = t_sne(feature_name, layer, "bad", 0)

    plt.scatter(l_x, l_y, c="green", s=12, label="good", marker="o", alpha=0.5)
    plt.scatter(p_x, p_y, c="red", s=12, label="bad", marker="x", alpha=0.5)
    plt.legend(loc='upper right')
    plt.title(layer)
    plt.show()


def t_sen_new(feature_name, layer, isNorm):
    all = get_data_new(feature_name, layer)
    print("cal....")
    fea = TSNE(n_components=2).fit_transform(all)
    if isNorm:
        fea = (fea - fea.min(0)) / (fea.max(0) - fea.min(0))
    print("printing....")
    plt.scatter(fea[:1680, 0], fea[:1680, 1], label="legitimate",  c="seagreen", s=19.5)
    #plt.scatter(fea[:7201, 0], fea[:7201, 1], label="legitimate", c="seagreen", s=19.5)
    #7201
    plt.scatter(fea[1680:, 0], fea[1680:, 1], label="malicious", c="orangered", s=19.5)
    #plt.scatter(fea[7201:, 0], fea[7201:, 1], label="malicious", c="orangered", s=19.5)

    # plt.scatter(fea[1000:, 0], fea[1000:, 1], label="bad")
    plt.legend(loc='upper right', fontsize='large')
    plt.tight_layout()
    #plt.savefig(layer + "_1000.bmp",format='bmp', dpi=1000)
    plt.savefig(layer + "1600_ALL", format='jpeg', dpi=600)
    plt.show()


def t_sne(feature_name, layer, label, isNorm):
    data, num, dim = get_data(feature_name, layer, label)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(data)
    if isNorm:
        data = (data - data.min(0)) / (data.max(0) - data.min(0))

    x = data[:, 0]
    y = data[:, 1]

    return x, y


def save_to_group(group, target, i, data):
    if target[i].data == 0:
        group['bad'].resize((group['bad'].shape[0] + 1), axis=0)
        group['bad'][-1:] = data[i]
    else:
        group['good'].resize((group['good'].shape[0] + 1), axis=0)
        group['good'][-1:] = data[i]
    pass

def get_feature(model_name, dim):

    # load developing data
    name = "CSIC"
    #dataset = "../req_vocab/%s_0.05_encoded_del.h5py" % (name)
    dataset = "../req_vocab/CSIC_test_v_encoded_del.h5py"
    #dataset = "CSIC_test_3000_encoded_del.h5py"
    train_dataset = EncodedURLDataset(dataset, "test")
    print(train_dataset.__len__())
    print("Transferring training data into iterator...")
    test_loader = DataLoader(train_dataset, batch_size=128, num_workers=0, drop_last=False, shuffle=False)
    w2v_model_name = "../w2v_models/CSIC_60_w4.m"
    model = CharCNN(dim, w2v_model_name, False)
    model.cuda()


    cnn_model_name = model_name
    # load status
    model_path = model_name
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    #f = h5py.File("CSIC_1000_60_feature_visual.h5py", "a")
    f = h5py.File("CSIC_v_60_feature_visual.h5py", "a")
    # g1 = f.create_group("conv1")
    # g2 = f.create_group("conv2")
    #g3 = f.create_group("conv3")
    #g4 = f.create_group("embedding")
    g5 = f.create_group("flatten")


    # g1.create_dataset('bad', shape=(1, 405504), maxshape=(None, 405504), chunks=True)
    # g1.create_dataset('good', shape=(1, 405504), maxshape=(None, 405504), chunks=True)
    #
    # g2.create_dataset('bad', shape=(1, 133120), maxshape=(None, 133120), chunks=True)
    # g2.create_dataset('good', shape=(1, 133120), maxshape=(None, 133120), chunks=True)

    # g3.create_dataset('bad', shape=(1, 41984), maxshape=(None, 41984), chunks=True)
    # g3.create_dataset('good', shape=(1, 41984), maxshape=(None, 41984), chunks=True)
    #
    # g4.create_dataset('bad', shape=(1, 24000), maxshape=(None, 24000), chunks=True)
    # g4.create_dataset('good', shape=(1, 24000), maxshape=(None, 24000), chunks=True)

    g5.create_dataset('bad', shape=(1, 39936), maxshape=(None, 39936), chunks=True)
    g5.create_dataset('good', shape=(1, 39936), maxshape=(None, 39936), chunks=True)

    # g4.create_dataset('bad', shape=(1, 6144), maxshape=(None, 6144), chunks=True)
    # g4.create_dataset('good', shape=(1, 6144), maxshape=(None, 6144), chunks=True)

    for i_batch, (data) in enumerate(test_loader):
        inputs, target = data
        #target.sub_(1)
        batch_size = target.shape[0]
        if 1:
            inputs, target = inputs.cuda(), target.cuda()

        inputs = Variable(inputs, volatile=True).type(torch.cuda.LongTensor)
        target = Variable(target).type(torch.cuda.LongTensor).reshape(-1)

        output = model(inputs)
        logit = output[0]
        embedding = output[1][0].cpu().detach().numpy()
        embedding = embedding.reshape(batch_size, -1)
        conv1 = output[1][1].cpu().detach().numpy()
        conv1 = conv1.reshape(batch_size, -1)
        #max1 = output[1][1].cpu().detach().numpy()
        #max1 = max1.transpose(1, 3)
        #max1 = max1.reshape(128, -1)
        conv2 = output[1][2].cpu().detach().numpy()
        conv2 = conv2.reshape(batch_size, -1)

        conv3 = output[1][3].cpu().detach().numpy()
        conv3 = conv3.reshape(batch_size, -1)

        # conv4 = output[1][3].cpu().detach().numpy()
        # conv4 = conv4.reshape(128, -1)
        flatten = output[1][4].cpu().detach().numpy()
        print("writing batch " + str(i_batch))
        for i in range(batch_size):
            #save_to_group(g1, target, i, conv1)
            #save_to_group(g2, target, i, conv2)
            #save_to_group(g3, target, i, conv3)
            #save_to_group(g4, target, i, embedding)
            save_to_group(g5, target, i, flatten)



if __name__ == '__main__':
    # prepare datalaoder

    model_path = "../trained_models/"
    model_name = model_path + "CSIC_60_w4.pth.tar"

    # get_feature(model_name, 60)
    feature_name = "CSIC_v_60_feature_visual.h5py"
    # feature_name = "CSIC_test_60_feature_visual.h5py"

    t_sen_new(feature_name, "flatten", False)

    #print(conv1)

