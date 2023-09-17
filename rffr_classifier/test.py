
import random
import numpy as np

import os
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

from tqdm import tqdm
from configs.config import config 
from models.model_detector import RFFRL

from utils.simple_evaluate import eval_one_dataset
from torch.utils.data import DataLoader
from utils.dataset import Deepfake_Dataset

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

# This is a stand-alone test-file.
# It loads in the deepfake detector (net.dd) and tests on the test set specified in "test_path".

net = RFFRL().cuda()

paths = ['']
for modelpath in paths:

    checkpoint = torch.load(modelpath)
    net.dpc.load_state_dict(checkpoint['state_dict'])

    for test_path in config.test_label_path:
        test_json = open('../data_label/' + test_path)
        print("Test dataset:", test_path)
        samples = json.load(test_json)
        total = len(samples)

        test_dataloader = DataLoader(Deepfake_Dataset(samples, train=False), batch_size=16, shuffle=False)
        _, acc, auc = eval_one_dataset(test_dataloader, net)

        print('Model: ', os.path.basename(modelpath))
        print('ACC:', round(acc,4))
        print('AUC:', round(auc,4) * 100)

        f = open('logs/evaluate.txt', 'a')
        f.write('\nModel: ' + os.path.basename(modelpath))
        f.write('\nTest path: ' + test_path)
        f.write('\nACC: ' + str(round(acc, 4)))
        f.write('\nAUC: ' + str(round(auc, 4)))
        f.write('\n')
        f.close()