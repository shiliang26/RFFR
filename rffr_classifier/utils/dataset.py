import os
import cv2
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
sys.path.append("..")
from configs.config import config


class Deepfake_Dataset(Dataset):
    def __init__(self, data_dict, train=True):

        if train:
            self.photo_path = [config.dataset_base + dicti['path'] for dicti in data_dict]
        else:
            self.photo_path = [dicti['path'] for dicti in data_dict]

        self.photo_label = [dicti['label']for dicti in data_dict]

        self.transform_1 = T.ToTensor()
        self.transform_2 = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            
    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):

        img_path = self.photo_path[item]
        label = self.photo_label[item]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        unnormed = self.transform_1(img)
        normed = self.transform_2(img)

        return normed, unnormed, label
