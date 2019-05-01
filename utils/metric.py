from termcolor import cprint, colored as c
from sklearn import metrics
def inc(d, label):
    if label in d:
        d[label] += 1
    else:
        d[label] = 1

def precision_recall(output, target):
    assert len(output) == len(target), "output len: {} != target len: {}".format(len(output), len(target))
    labels = set(target)
    TP = {}
    FP = {}
    TP_plus_FN = {}
    TP_plus_FP = {}
    FP_plus_TN = {}
    for i in range(len(output)):

        inc(TP_plus_FN, target[i])
        inc(TP_plus_FP, output[i])
        if target[i] == output[i]:
            inc(TP, output[i])
        if target[i] != output[i]:
            inc(FP, output[i])

    for label in labels:
        if label not in TP_plus_FN:
            TP_plus_FN[label] = 0
        if label not in TP_plus_FP:
            TP_plus_FP[label] = 0

    precision = {label: 0. if TP_plus_FP[label] == 0 else ((TP[label] if label in TP else 0) / float(TP_plus_FP[label])) for label in labels}
    recall = {label: 0. if TP_plus_FN[label] == 0 else ((TP[label] if label in TP else 0) / float(TP_plus_FN[label])) for label in labels}
    #FPR = {label: 0. if FP_plus_TN[label] ==0 else }
    return precision, recall, TP, TP_plus_FN, TP_plus_FP


def F_score(p, r):

    f_scores = {
        label: None if p[label] == 0 and r[label] == 0 else (0 if p[label] == 0 or r[label] == 0 else 2 / (1 / p[label] + 1 / r[label]))
        for label in p
    }
    return f_scores

def write_cases(predicates_all, target_all):
    FP = 0
    TN = 0
    TP = 0
    FN = 0

    f_FN = open("corpus/1000_FN.txt", "a")
    f_FP = open("corpus/1000_FP.txt", "a")
    f_TP = open("corpus/1000_TP.txt", "a")
    f_TN = open("corpus/1000_TN.txt", "a")
    for i in range(len(predicates_all)):
        if predicates_all[i] == 1 and target_all[i] == 0:
            FN += 1

        if predicates_all[i] == 0 and target_all[i] == 0:
            TP += 1
        if predicates_all[i] == 1 and target_all[i] == 1:
            TN += 1
        if predicates_all[i] == 0 and target_all[i] == 1:
            FP += 1
    print("TP/FP/TN/FN")
    print(TP, FP, TN, FN)

def cal_metric_all(predicates_all, target_all):
    FP = 0
    TN = 0
    TP = 0
    FN = 0
    # for i in range(len(predicates_all)):
    #     if predicates_all[i] == 1 and target_all[i] == 0:
    #         FP += 1
    #     if predicates_all[i] == 0 and target_all[i] == 0:
    #         TN += 1
    #     if predicates_all[i] == 1 and target_all[i] == 1:
    #         TP += 1
    #     if predicates_all[i] == 0 and target_all[i] == 1:
    #         FN += 1
    for i in range(len(predicates_all)):
        if predicates_all[i] == 1 and target_all[i] == 0:
            FN += 1
        if predicates_all[i] == 0 and target_all[i] == 0:
            TP += 1
        if predicates_all[i] == 1 and target_all[i] == 1:
            TN += 1
        if predicates_all[i] == 0 and target_all[i] == 1:
            FP += 1
    print("TP/FP/TN/FN")
    print(TP, FP, TN, FN)

    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    recall = TP / (TP + FN)
    try:
        F1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        F1 = 0
    #auc = metrics.auc(FPR, TPR)
    fpr, tpr, thresholds = metrics.roc_curve(target_all, predicates_all, pos_label=2)
    auc = metrics.auc(fpr, tpr)

    return FPR, TPR, precision, F1, auc




def print_f_score(output, target):
    """returns: 
        p<recision>, 
        r<ecall>, 
        f<-score>, 
        {"TP", "p", "TP_plus_FP"} """
    p, r, TP, TP_plus_FN, TP_plus_FP = precision_recall(output, target)
    f = F_score(p, r)

    # cprint("Label: " + c(("  " + str(10))[-5:], 'red') +
    #            "\tPrec: " + c("  {:.1f}".format(0.335448 * 100)[-5:], 'green') + '%' +
    #            " ({:d}/{:d})".format(1025, 1254).ljust(14) +
    #            "Recall: " + c("  {:.1f}".format(0.964 * 100)[-5:], 'green') + "%" +
    #            " ({:d}/{:d})".format(15, 154).ljust(14) +
    #            "F-Score: " +  (c("  {:.1f}".format(0.5 * 100)[-5:], "green") + "%")
    #            )

    for label in f.keys():
        cprint("Label: " + c(("  " + str(label))[-5:], 'red') +
               "\tPrec: " + c("  {:.1f}".format(p[label] * 100)[-5:], 'green') + '%' +
               " ({:d}/{:d})".format((TP[label] if label in TP else 0), TP_plus_FP[label]).ljust(14) +
               " TPR: " + c("  {:.1f}".format((r[label] if label in r else 0) * 100)[-5:], 'green') + "%" +
               " ({:d}/{:d})".format((TP[label] if label in TP else 0), TP_plus_FN[label]).ljust(14) +
               " F-Score: " + ("  N/A" if f[label] is None else (c("  {:.1f}".format(f[label] * 100)[-5:], "green") + "%"))
               )
    # return p, r, f, _


if __name__ == '__main__':

    import torch
    import torch.autograd as autograd
    #output = [1,1,1,1,1,2,0,2,2,2,2]
    output = [0,1,1,0]
    output = torch.LongTensor(output)
    # target = [0,0,2,1,2,2,1,2,1,2,0]
    target = [0,0,1,1]
    #target = [1,3,2,3,3,3,3,3,0,3,3]

    target = torch.LongTensor(target)
    output = autograd.Variable(output)
    target = autograd.Variable(target)
    print('output')
    print(output.data.numpy().tolist())
    print('target')
    print(target.data.numpy().tolist())

    
    precision, recall, TP, TP_plus_FN, TP_plus_FP = precision_recall(output.data.numpy().tolist(), target.data.numpy().tolist())
    print('precision')
    print(precision)
    print('recall')
    print(recall)
    print('TP')
    print(TP)
    print('TP_plus_FN')
    print(TP_plus_FN)
    print('TP_plus_FP')
    print(TP_plus_FP)
    # print(dic)


    f_scores = F_score(precision, recall)
    print('f_scores')
    print(f_scores)
    # print(f_scores.keys())

    
    print('\r')
    print_f_score(output.data.numpy().tolist(), target.data.numpy().tolist())