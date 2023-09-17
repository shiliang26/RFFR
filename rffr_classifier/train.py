
import os
import random
import hashlib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from configs.config import config
from datetime import datetime
import time

from utils.utils import save_checkpoint, AverageMeter, Logger
from utils.utils import accuracy, mkdirs, save_code, remove_used
from utils.simple_evaluate import eval_multiple_dataset
from utils.get_loader import get_dataset

from models.model_detector import RFFRL

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

def train():
    # Profile
    timenow = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    runhash = hashlib.sha1()
    runhash.update(timenow.encode('utf-8'))
    runhash.update(config.comment.encode('utf-8'))
    runhash = runhash.hexdigest()[:6]
    save_code(timenow, runhash)
    mkdirs(timenow + '_' + runhash)

    # Model
    net = RFFRL()
    net = net.cuda()
    optimizer = optim.Adam(net.dd.parameters(), lr=config.lr)
    criterion = {'softmax': nn.CrossEntropyLoss().cuda()}

    # Evaluators
    aucs = [0] * 4
    best_aucs = [0] * 4
    is_best_AUC = [False] * 4
    classifer_top1 = AverageMeter()

    # Data
    train_real_dataloader, train_fake_dataloader, test_dataloader = get_dataset()
    train_real_iter = iter(train_real_dataloader)
    train_real_iters_per_epoch = len(train_real_iter)
    train_fake_iter = iter(train_fake_dataloader)
    train_fake_iters_per_epoch = len(train_fake_iter)

    # Logs
    log = Logger()
    log.initialize(timenow, runhash)

    # Training
    epoch = 0
    beginning_iter = 0
    for iter_num in range(beginning_iter, config.max_iter+1):
        if (iter_num % train_real_iters_per_epoch == 0):
            train_real_iter = iter(train_real_dataloader)
        if (iter_num % train_fake_iters_per_epoch == 0):
            train_fake_iter = iter(train_fake_dataloader)
        if (iter_num != 0 and iter_num % config.iter_per_epoch == 0):
            epoch = epoch + 1
            classifer_top1.reset()
            if epoch % 2 == 0:
                remove_used(os.path.join(config.best_model_path, timenow + '_' + runhash))
            
        net.train(True)
        optimizer.zero_grad()

        img_real, img_real_unnormed, label_real = train_real_iter.next()
        img_real = img_real.cuda()
        img_real_unnormed = img_real_unnormed.cuda()
        label_real = label_real.cuda()

        img_fake, img_fake_unnormed, label_fake = train_fake_iter.next()
        img_fake = img_fake.cuda()
        img_fake_unnormed = img_fake_unnormed.cuda()
        label_fake = label_fake.cuda()

        input_data = torch.cat([img_real, img_fake], dim=0)
        input_unnormed = torch.cat([img_real_unnormed, img_fake_unnormed], dim=0)
        input_label = torch.cat([label_real, label_fake], dim=0)

        _, classifier_out = net(input_unnormed, input_data)
        
        cls_loss = criterion["softmax"](classifier_out, input_label)
        optimizer.zero_grad()
        cls_loss.backward()
        optimizer.step()

        acc = accuracy(classifier_out, input_label, topk=(1,))
        classifer_top1.update(acc[0])

        log.write_evaluations(epoch, iter_num, aucs, best_aucs, classifer_top1.avg, per_iteration=True)
            
        if (iter_num != 0 and (iter_num+1) % config.iter_per_eval == 0):

            aucs = eval_multiple_dataset(test_dataloader, net)

            for i in range(4):
                is_best_AUC[i] = False
                if aucs[i] > best_aucs[i]:
                    best_aucs[i] = aucs[i]
                    is_best_AUC[i] = True
            
            if epoch % 100 == 0 or is_best_AUC[0] or is_best_AUC[1] or is_best_AUC[2] or is_best_AUC[3]:
                save_list = [epoch, aucs, best_aucs]
                save_checkpoint(save_list, is_best_AUC, net, timenow + '_' + runhash)

            log.write_evaluations(epoch, iter_num, aucs, best_aucs, classifer_top1.avg, per_iteration=False)
            log.write('\n')
            time.sleep(0.01)
        
if __name__ == '__main__':
    train()
