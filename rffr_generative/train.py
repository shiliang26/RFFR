
from utils import save_checkpoint, AverageMeter, Logger
from utils import mkdirs, time_to_str, save_code
from get_data import get_cdf
from dataset import Deepfake_Dataset
from torch.utils.data import DataLoader

import json
import random
import hashlib
import numpy as np
from configs.config import config
from datetime import datetime
import time
from timeit import default_timer as timer

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import timm.optim.optim_factory as optim_factory

from models.model_mae import mae_vit_base_patch16

from tensorboardX import SummaryWriter

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train():

    mkdirs()
    
    timenow = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    runhash = hashlib.sha1()
    runhash.update(timenow.encode('utf-8'))
    runhash.update(config.comment.encode('utf-8'))
    runhash = runhash.hexdigest()[:6]
    
    save_code(timenow, runhash)

    train_dict = get_cdf()
    length = len(train_dict)
    train_data = train_dict[:-int(0.001 * length)]
    test_data = train_dict[-int(0.001 * length):]

    real_dataloader = DataLoader(Deepfake_Dataset(train_data, train=True, real=True), batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last = True)
    test_dataloader = DataLoader(Deepfake_Dataset(test_data, train=True, real=True), batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last = True)

    best_loss = np.inf
    epoch = 0
    start_iter = 0

    loss_logger = AverageMeter()
    eval_loss_logger = AverageMeter()

    net = mae_vit_base_patch16().cuda()
    if config.in_pretrained is not None:
        checkpoint = torch.load(config.in_pretrained)
        net.load_state_dict(checkpoint["model"], strict=True)
    if(len(config.gpus) > 1):
        net = torch.nn.DataParallel(net).cuda()

    # net = torch.compile(net)
    # torch.set_float32_matmul_precision('high')
    
    param_groups = optim_factory.param_groups_weight_decay(net, config.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=config.lr, betas=(config.beta1, config.beta2))

    if config.in_pretrained is None and config.my_pretrained is not None:
        checkpoint = torch.load(config.my_pretrained)
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint['adam_dict'])
        epoch = checkpoint["epoch"]
        start_iter = epoch * config.iter_per_epoch


    if(config.enable_tensorboard):
        tblogger = SummaryWriter(comment=config.comment)
    
    log = Logger()
    log.open(config.logs + timenow + '_' + runhash + '_' + config.comment + '.txt', mode='a')
    log.write("\n-------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-------------------'))
    log.write('Random seed: %d\n' % config.seed)
    log.write('Comment: %s\n' % config.comment)
    log.write('** start training target model! **\n')
    log.write(
        '--------|----- Train -----|---- Best ----|---- Test ----|\n')
    log.write(
        '  iter  |      loss       |     loss     |     loss     |\n')
    log.write(
        '--------------------------------------------------------|\n')
    start = timer()
    criterion = {
        'mse': nn.MSELoss().cuda(),
        'l1': nn.L1Loss().cuda(),
        'softmax': nn.CrossEntropyLoss().cuda()
    }

    iter_per_epoch = config.iter_per_epoch # iters that the model need to be tested
    max_iter = config.max_iter

    train_real_iter = iter(real_dataloader)
    train_real_iters_per_epoch = len(train_real_iter)

    for iter_num in range(start_iter, max_iter+1):
        if iter_num % train_real_iters_per_epoch == 0:
            train_real_iter = iter(real_dataloader)
        if iter_num != 0 and iter_num % iter_per_epoch == 0:
            epoch = epoch + 1
            loss_logger.reset()

        # Learning rate schedule
        if iter_num < config.lr_warmup:    # 10% Time
            lr = config.lr * ((iter_num + 1) / config.lr_warmup)
        else:
            lr = config.lr * 0.5 * (1 + np.cos(np.pi * (iter_num - config.lr_warmup) / (config.max_iter - config.lr_warmup)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        net.train(True)

        img_real = next(train_real_iter)
        img_real = img_real.cuda()
        input_data = img_real

        rec_loss, pred, mask = net(input_data)
        rec_loss = rec_loss.mean()
        
        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()
        
        loss_logger.update(rec_loss.item())
        
        print('\r', end='', flush=True)
        print('  %4.1f |     %6.4f      |     %6.4f    |     %6.4f    |    %s'
            % (
                epoch + (iter_num % iter_per_epoch) / iter_per_epoch,
                loss_logger.avg,
                best_loss,
                eval_loss_logger.avg,
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)

        
        if (iter_num != 0 and (iter_num+1) % config.iter_per_epoch == 0):

            if loss_logger.avg < best_loss:
                is_best_loss = True
                best_loss = loss_logger.avg
                current_loss = loss_logger.avg
            
            eval_loss_logger.reset()

            for test_data in test_dataloader:
                test_data = test_data.cuda()
                rec_loss, _, _ = net(test_data)
                rec_loss = rec_loss.mean().cpu().detach().numpy()
                eval_loss_logger.update(rec_loss)
            
            if ((epoch + 1) % 50 == 0):
                save_list = [epoch + 1, current_loss]
                save_checkpoint(save_list, net, optimizer)
            
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f |     %6.4f      |     %6.4f    |     %6.4f    |   %s'
                % (
                epoch + 1,
                loss_logger.avg,
                best_loss,
                eval_loss_logger.avg,
                time_to_str(timer() - start, 'min')))
            log.write('\n')

            if(config.enable_tensorboard):
                info = {
                'Loss_train': loss_logger.avg,
                'lowest_loss': best_loss,
                'Test_Loss': eval_loss_logger.avg,
                }
                for tag, value in info.items():
                    tblogger.add_scalar(tag, value, epoch)

        
if __name__ == '__main__':
    train()
