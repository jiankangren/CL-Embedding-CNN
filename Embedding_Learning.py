import gensim
import numpy as np
import h5py

class CharacterEmbedding(object):
    def __init__(self):
        pass

    def createCharacterEmbedding(self, corpus_file, dim):
        corpus_l_p = self.loadData(corpus_file)
        corpus = corpus_l_p

        # create and save skip-gram model
        # model = gensim.models.Word2Vec(corpus, window=4, min_count=1, size=dim, sg=1)
        # model.save('../../models/req_csic_c2v_%s.m' % dim)

        # load trained skip-gram model
        model = gensim.models.Word2Vec.load("model/req_csic_c2v_%s.m" % dim)
        return model

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
        return corpus


    def makeVect(self, str, str_len, vec_size, model):
        if str == ' ':
            char_vec = np.zeros((str_len, vec_size))

        elif len(str) < str_len:
            char_list = self.splitCharacter(str)
            char_vec = model.wv[char_list]
            char_vec = np.vstack((char_vec, np.zeros((str_len - len(str), vec_size))))

        else:
            char_list = self.splitCharacter(str)[:str_len]
            char_vec = model.wv[char_list]
        return char_vec


    def reqToVect(self, readfile, label, model, dim, req_len, phase):
        """
        :param readfile:
        :param label: 1 for malicious, 0 for legi
        :param model: path to skip-gram model
        :param dim: dimension fo character embedding
        :param phase: "train" or "test"
        :return:
        """
        fr = open(readfile, 'r')
        savefile = "../vectors/%s_csic_%s_%s.hdf5" % (phase, req_len, dim)

        f = h5py.File(savefile, "a")
        if label == 0:
            f.create_dataset('x_train', shape=(1, req_len, dim), maxshape=(None, req_len, dim), chunks=True)
            f.create_dataset('y_train', shape=(1, 1), maxshape=(None, 1))

        for line in fr.readlines():
            print(line)
            line = line.strip()
            if len(line) == 0 : continue
            else:
                req_vec = self.makeVect(line, req_len, dim, model)
                f['x_train'].resize((f['x_train'].shape[0] + 1), axis=0)
                f['x_train'][-1:] = req_vec
                f['y_train'].resize((f['y_train'].shape[0] + 1), axis=0)
                f['y_train'][-1:] = label
        f.close()
        fr.close()

    def createModel(self, dim, corpusfile, readfile_l, readfile_p, req_len, phase):
        """
        :param dim: dimension of character embedding
        :param corpusfile: cropus path
        :param readfile_l: legi requests  path
        :param readfile_p: malicious requests  path
        :param phase: "train" of "test"
        :return:
        """
        model = self.createCharacterEmbedding(corpusfile, dim)

        self.reqToVect(readfile_p, 0, model, dim, req_len, phase)
        self.reqToVect(readfile_l, 1, model, dim, req_len, phase)
        print('done')


    def splitCharacter(self, str):
        l = []
        for s in str:
            l.extend(s)
        return l

if __name__ == "__main__":

    corpus = "/path to/cropus_all.txt"
    legi = "dataset/WAF queries dataset/WAF_Legi-0.75.txt"
    mali = "dataset/WAF queries dataset/WAF_Malicous-0.75.txt"
    req_len = 100  # 100 for WAF queries dataset, 400 for HTTP CSIC
    dim = 70
    charEmbedding = CharacterEmbedding()
    charEmbedding.createModel(dim, corpus, legi, mali, req_len, "train")



