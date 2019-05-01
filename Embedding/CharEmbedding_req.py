import h5py

import torch
import numpy as np
import json

import gensim
from torch import nn
import pandas as pd
def splitCharacter(str):
    l = []
    for s in str:
        l.extend(s)
    return l

def build_vocab(corpus):
    # 给每个单词编码，也就是用数字来表示每个单词，这样才能够传入word embeding得到词向量。
    vocab = set(corpus)  # 通过set将重复的单词去掉
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}

    word_to_idx['<pad>'] = 0
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<pad>'
    return word_to_idx, idx_to_word

def get_chars(corpus_path):
    corpus = []
    with open(corpus_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            chars = splitCharacter(line)
            chars = set(chars)
            corpus.extend(chars)
    corpus = set(corpus)
    return corpus

def splitUrl(url):
    import tldextract
    from urllib.parse import urlsplit
    extr = tldextract.extract(url)
    #         pro=urllib.parse.urlsplit(url)
    pro = urlsplit(url)
    return pro.scheme, extr.subdomain, extr.domain, extr.suffix, pro.path

def pad_str(str, length):
    if len(str) < length:
        for _ in range(length - len(str)):
            str.append('<pad>')
    else:
        str = str[:length]
    return str

def encode_Str(name, str):
    with open("../req_vocab/" + name + "_w2i.json", "r") as f:
        w2i = json.load(f)
    encoded_str = [w2i[c] for c in str]
    return encoded_str




def save_to_group(group, label, data):
    if label == 0:
        group['x'].resize((group['x'].shape[0] + 1), axis=0)
        group['x'][-1:] = data
        group['y'].resize((group['y'].shape[0] + 1), axis=0)
        group['y'][-1:] = 0
    else:
        group['x'].resize((group['x'].shape[0] + 1), axis=0)
        group['x'][-1:] = data
        group['y'].resize((group['y'].shape[0] + 1), axis=0)
        group['y'][-1:] = 1


def load_word_vectors(filename):
    w2v = gensim.models.Word2Vec.load(filename)
    i2w = w2v.wv.index2word
    i2w = {i: i2w[i] for i in range(len(i2w))}
    i2w[len(i2w)] = "<pad>"
    w2i = {value: key for key, value in i2w.items()}
    vectors = w2v.wv.syn0
    return i2w, w2i, vectors


def create_emb_layer(weights_matrix, trainable=False):
    num_embeddings = weights_matrix.shape[0]
    embedding_dim = weights_matrix.shape[1]
    weights = torch.FloatTensor(weights_matrix)
    # emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    # emb_layer.load_state_dict({'weight': weights_matrix})
    emb_layer = nn.Embedding.from_pretrained(weights)
    if trainable:
        emb_layer.weight.requires_grad = True
    else:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim


class CharEmbedding():
    def __init__(self, dim, w2v_model_path):
        self.dim = dim
        # self.w2v_model_path = w2v_model_path
        # self.w2i = build_vocab(corpus)
        #self.i2w = {v: k for k, v in self.w2i.items()}
        self.index_to_word, self.word_to_index, self.word_vectors = load_word_vectors(w2v_model_path)
        self.model_path = w2v_model_path

        self.courpus = "../corpus/CSIC_corpus.txt"



    def create_w2v_model(self):
        corpus = self.loadData(self.courpus)
        model = gensim.models.Word2Vec(corpus, window=4, min_count=1, size=self.dim, sg=1)
        model.save('../w2v_models/CSIC_%s_w4.m' % self.dim)
        return model

    def splitCharacter(self, str):
        l = []
        for s in str:
            l.extend(s)
        return l

    def loadData(self, file):
        corpus = []
        fr = open(file, 'r')
        i = 0
        for line in fr.readlines():
            line.strip()
            if line == '':
                print('line is null')
                continue
            corpus.append(self.splitCharacter(line))
            i += 1
        fr.close()
        return corpus

    def get_i2w(self):
        return self.index_to_word

    def get_embedding(self, word):
        return self.word_vectors[self.word_to_index[word]]

    def create_emb_layer(self, trainable=False):
        num_embeddings = self.word_vectors.shape[0]
        embedding_dim = self.word_vectors.shape[1]
        self.word_vectors
        weights = torch.FloatTensor(self.word_vectors)
        weights = torch.cat((weights, torch.zeros((1, embedding_dim))), 0)
        emb_layer = nn.Embedding.from_pretrained(weights)
        if trainable:
            emb_layer.weight.requires_grad = True
        else:
            emb_layer.weight.requires_grad = False
        return emb_layer

    def pre_process_URL(self, path, data):
        URL_path = path + data

        phase = data.split("_")[0]
        label = data.split("_")[1]
        name = data.split(".")[0].split("_")[2]

        if label == "bad":
            label = 0
        else:
            label = 1


        f = h5py.File("../data/" + name + "_encoded.h5py", "a")
        if name == "CSIC":
            req_legth = 400
        else:
            req_legth = 100

        if phase == "train" and label == 0:
            g1 = f.create_group("train")
            g2 = f.create_group("val")
            g3 = f.create_group("test")

            g1.create_dataset('x', shape=(1, req_legth), maxshape=(None, req_legth), data=None, chunks=True)
            g1.create_dataset('y', shape=(1, 1), maxshape=(None, 1), data=None, chunks=True)

            g2.create_dataset('x', shape=(1, req_legth), maxshape=(None, req_legth), data=None, chunks=True)
            g2.create_dataset('y', shape=(1, 1), maxshape=(None, 1), data=None, chunks=True)

            g3.create_dataset('x', shape=(1, req_legth), maxshape=(None, req_legth), data=None, chunks=True)
            g3.create_dataset('y', shape=(1, 1), maxshape=(None, 1), data=None, chunks=True)
        print("Starting encoding...")
        with open(URL_path, 'r') as fr:
            for line in fr:
                line = line.strip()
                if len(line) == 0:
                    continue
                else:
                    line_chars = splitCharacter(line)
                    line_chars = pad_str(line_chars, req_legth)
                    #encoded_URL = encode_Str(name, URL_chars)
                    encoded_URL = [self.word_to_index[c] for c in line_chars]

                    if phase == "train":
                        save_to_group(f["train"], label, encoded_URL)
                    elif phase == "val":
                        save_to_group(f["val"], label, encoded_URL)
                    else:
                        save_to_group(f["test"], label, encoded_URL)
        f.close()
        print("Finish..")


def del_first_row(f, name):
    f = h5py.File("../data/" + name + "_encoded.h5py", "r")
    f2 = h5py.File(
        "../data/" + name + "_encoded_del.h5py", "a")

    if name == "CSIC":
        req_legth = 400
    else:
        req_legth = 100

    for p in ["train", "test"]:
        for i in ["x", "y"]:
            x_train = f[p][i]
            t = x_train[1:, :]

            if i == "x":
                f2[p].create_dataset(i, data=t, maxshape=(None, req_legth), chunks=True)
            else:
                f2[p].create_dataset(i, data=t, maxshape=(None, 1), chunks=True)
    f2.close()
    f.close()

if __name__ == "__main__":

    path  = "../corpus/"
    data = ["train_bad_CSIC.txt", "train_good_CSIC.txt", "test_good_CSIC.txt", "test_bad_CSIC.txt"]
    # Building vocab
    w2v_model_path = "../w2v_models/CSIC_60_w4.m"
    charEmbedding = CharEmbedding(60, w2v_model_path)
    for f in data:
         charEmbedding.pre_process_URL(path, f)


    # delete the first zeros row
    del_first_row("CSIC")




