
import cv2
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
sys.path.append("..")
from configs.config import config


class Deepfake_Dataset(Dataset):
    def __init__(self, paths, transforms=None, train=True):
        self.train = train
        self.paths = paths
        
        self.transforms = T.Compose([
            T.ToTensor()
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            #             std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):

        img_path = self.paths[item]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)

        return img
        