from torch.autograd import Variable
import torch
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def eval_one_dataset(valid_dataloader, model):
    
    model.eval()

    prob_list = []
    label_list = []

    for input, unnormed, label in valid_dataloader:
        input = Variable(input).cuda()
        unnormed = Variable(unnormed).cuda()

        _, cls_out = model(unnormed, input, test=True)
        prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
        
        prob_list = np.append(prob_list, prob)
        label_list = np.append(label_list, label)

    auc_score = roc_auc_score(label_list, prob_list)

    return auc_score

def eval_multiple_dataset(dataloaders, model):
    aucs = []
    for dataloader in dataloaders:
        auc = eval_one_dataset(dataloader, model)
        aucs.append(auc)
    return aucs