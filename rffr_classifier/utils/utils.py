import torch
import os
import sys
from configs.config import config
import shutil
from timeit import default_timer as timer

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

def mkdirs(runhash):
    if not os.path.exists(config.best_model_path + runhash):
        os.makedirs(config.best_model_path + runhash)
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

    def initialize(self, timenow, runhash):
        self.open(config.logs + timenow + '_' + runhash + '_' + config.comment + '.txt', mode='a')
        self.write('\n %s [START %s] %s\n\n' % ('-' * 50, timenow, '-' * 50))
        self.write('Run Hash: %s\n' % runhash)
        self.write('Random seed: %d\n' % config.seed)
        self.write('Comment: %s\n' % config.comment)
        self.write('** start training target model! **\n')


        self.write('--------|------------- V-AUC --------------|--- Train ---|------------Best AUC ------------|--------------|\n')
        self.write('  iter  |     A      B       C       D     |    top-1    |     A      B       C       D    |    time      |\n')

        self.write('%s|\n' % ('-' * 105))
        self.time_start = timer()

    def write_evaluations(self, epoch, iter_num, aucs, best_aucs, train_loss, per_iteration):

        is_file = not per_iteration

        self.write('\r', is_file=is_file)
        self.write(
            '  %4.1f  |  %6.4f  %6.4f  %6.4f  %6.4f  |   %6.4f   |   %6.4f  %6.4f  %6.4f  %6.4f  |    %s'
            % (
                epoch + (iter_num % config.iter_per_epoch) / config.iter_per_epoch,
                aucs[0], aucs[1], aucs[2], aucs[3],
                train_loss, 
                best_aucs[0], best_aucs[1], best_aucs[2], best_aucs[3],
                time_to_str(timer() - self.time_start, 'min')), is_file=is_file)


def save_checkpoint(save_list, is_best_AUC, model, runhash):
    epoch = save_list[0]
    aucs = save_list[1]
    best_aucs = save_list[2]
    state = {
        "epoch": epoch,
        "state_dict": model.dd.state_dict(),
        "aucs": aucs,
        "best_aucs": best_aucs
    }
    
    for i in range(4):
        if is_best_AUC[i]:
            filepath = config.best_model_path + runhash + '/' + '_'.join([str(i + 1), "_AUC", str(round(best_aucs[i], 5)), str(epoch)]) +  '.pth.tar'
            torch.save(state, filepath)
    if epoch % 100 == 0:
        filepath = config.best_model_path + runhash + '/' + '_'.join(['0', "_AUC", str(epoch)]) +  '.pth.tar'
        torch.save(state, filepath)


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


def remove_used(dir):
    files = os.listdir(dir)
    keep = []
    for i in range(5):
        group = [file for file in files if file[0] == str(i)]
        if len(group) == 0:
            continue
        keep.append(max(group))
    for file in files:
        if file[0] != '0' and file not in keep:
            # print("Removing ", file)
            filepath = os.path.join(dir, file)
            os.remove(filepath)