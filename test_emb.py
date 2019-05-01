import os
import argparse
import datetime
import sys
import errno
from model import CharCNN

from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils.metric import print_f_score, cal_metric_all
from utils.mydatasets import H5Dataset, EncodedURLDataset
from CNN_models.model_CharCNN_embedding import CharCNN as Char_Embedding
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='Character level CNN text classifier testing',
                                 formatter_class=argparse.RawTextHelpFormatter)
# model
parser.add_argument('--model-path', default='best_models/CharCNN_best_99.688.pth.tar',
                    help='Path to pre-trained acouctics model created by DeepSpeech training')
parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('--kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
#
parser.add_argument('--test-path', metavar='DIR',
                    help='path to testing data csv', default='data/ag_news_csv/test.csv')
parser.add_argument('--batch-size', type=int, default=256, help='batch size for training [default: 128]')
parser.add_argument('--alphabet-path', default='alphabet.json', help='Contains all characters for prediction')
# device
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--cuda', action='store_true', default=True, help='enable the gpu')
# logging options
parser.add_argument('--save-folder', default='req_test_result/', help='Location to save epoch models')
args = parser.parse_args()

if __name__ == '__main__':

    # with open(os.path.join('best_models/req_test_result/', 'req_test_result.csv'), 'w') as r:
    #     r.write('{:s},{:s},{:s},{:s},{:s}, {:s}, {:s}'.format('model', 'loss', 'acc', 'tpr', 'fpr',
    #                                                                       'precision', 'f1'))

    #w2v_model_name = "%s_%s_w4.m" % (name, str(args.num_features))
    #args.w2v_name = w2v_model_name


    args.best_model_path = "models_CharCNN/2019-04-29_10-38-54/CSIC_80_w4.pth.tar"
    model_name = args.best_model_path.split("/")[2].split(".")[0]
    #name = "CSIC"
    name = args.best_model_path.split("/")[2].split("_")[0]
    args.num_features = args.best_model_path.split("/")[2].split("_")[1]
    dataset = "fsys/%s_encoded_del.h5py" % (name)
    #w2v_model_path = "w2v_models/" + args.model_path.split("/")[2].split(".")[0] + ".m"
    w2v_model_path = "w2v_models/CSIC_60_w4.m"


    # load developing data
    print("\nLoading developing data...")

    test_dataset = EncodedURLDataset(dataset, "test")
    print("Transferring developing data into iterator...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, drop_last=False)


    print('\nNumber of testing samples: ' + str(test_dataset.__len__()))

    model = Char_Embedding(args.num_features, w2v_model_path)
    print("=> loading weights from '{}'".format(args.best_model_path))
    assert os.path.isfile(args.best_model_path), "=> no checkpoint found at '{}'".format(args.best_model_path)
    checkpoint = torch.load(args.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    # using GPU
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    print('\nTesting...')
    for i_batch, (data) in enumerate(test_loader):
        inputs, target = data
        #target.sub_(1)
        size += len(target)
        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        inputs = Variable(inputs, volatile=True).type(torch.cuda.LongTensor)
        target = Variable(target).type(torch.cuda.LongTensor).reshape(-1)
        logit = model(inputs)
        predicates = torch.max(logit, 1)[1].view(target.size()).data
        accumulated_loss += F.nll_loss(logit, target, size_average=False).data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        predicates_all += predicates.cpu().numpy().tolist()
        target_all += target.data.cpu().numpy().tolist()

    avg_loss = accumulated_loss / size
    # accuracy = corrects / size
    corrects = corrects.float()
    accuracy = 100.0 * corrects / size
    corrects = corrects.int()
    FPR, TPR, precision, f1, auc = cal_metric_all(predicates_all, target_all)

    print(
        '\nEvaluation - loss: {:.6f}  acc: {:.4f}% ({}/{}) TPR: {:.4f}% FPR: {:.4f}% prec: {:.4f}% '.format(
            avg_loss,
            accuracy,
            corrects,
            size, TPR * 100, FPR * 100, precision * 100))
    print_f_score(predicates_all, target_all)
    torch.cuda.empty_cache()
    #model_name = args.model_path.split('/')[7]
    with open(os.path.join('req_test_result/', 'CSIC_test_result.csv'), 'a') as r:
        r.write(
            '\n{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{}'.format(model_name, avg_loss, accuracy, TPR*100, FPR*100, precision*100,
                                                                                f1*100, auc, args.num_features))


