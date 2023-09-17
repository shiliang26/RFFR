import json
import math
import pandas as pd
import numpy as np
import torch
import os
import sys
from glob import glob
from configs.config import config
import shutil
from tqdm import tqdm
import random

def output_to_patch(output, row, col):
    # output: [b, c, h, w]
    # row / col:  [b, 1]
    W, H = output.shape[-1], output.shape[-2]
    offsets = (torch.cat([col, row], dim=1) * 16 + 16 * 1.5).cuda()
    h, w = 48, 48
    xs = (torch.arange(0, w) - (w - 1) / 2.0).cuda()
    ys = (torch.arange(0, h) - (h - 1) / 2.0).cuda()

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2

    offsets_grid = offsets[:, None, None, :] + grid[None, ...]

    # normalised grid  to [-1, 1]
    offsets_grid = (
        offsets_grid - offsets_grid.new_tensor([W/2, H/2])) / offsets_grid.new_tensor([W/2, H/2])

    return torch.nn.functional.grid_sample(output, offsets_grid, mode='bilinear', padding_mode='zeros', align_corners=None)

def calc_auc(y_labels, y_scores):
    # y_scores = y_scores / max(y_scores)
    f = list(zip(y_scores, y_labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i] == 1]
    pos_cnt = np.sum(y_labels == 1)
    neg_cnt = np.sum(y_labels == 0)
    auc = (np.sum(rankList) - pos_cnt*(pos_cnt+1)/2) / (pos_cnt*neg_cnt)
    return auc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mkdirs():
    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)
    if not os.path.exists(config.best_model_path):
        os.makedirs(config.best_model_path)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)
    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def save_checkpoint(save_list, model, optimizer, filename='_checkpoint.pth.tar'):
    epoch = save_list[0]
    current_loss = save_list[1]
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "adam_dict": optimizer.state_dict()
        }
          
    filepath = config.checkpoint_path + filename
    torch.save(state, filepath)
    
    shutil.copy(filepath, config.best_model_path + 'best_loss_' + str(round(current_loss, 5)) + '_' + str(epoch) + '.pth.tar')


def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def save_code(time, runhash):
    directory = '../history/' + time + '_' + runhash + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filepath in config.save_code:
        shutil.copy(filepath, directory + filepath.split('/')[-1])


