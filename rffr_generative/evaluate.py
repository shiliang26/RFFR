
from dataset import Deepfake_Dataset
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import json
from configs.config import config
# from Pytorch_UNet.unet.unet_model import UNet_backbone
from models.model_mae import mae_vit_base_patch16

import matplotlib.pyplot as plt

model_to_test = ''
test_json = open('../data_label/' + config.val_label_path)
test_dict = json.load(test_json)
dataloader = DataLoader(Deepfake_Dataset(test_dict, cali=16), batch_size=1, shuffle=True)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus


net = mae_vit_base_patch16()
net = nn.DataParallel(net).cuda()
checkpoint = torch.load(model_to_test)
net.load_state_dict(checkpoint["state_dict"], strict=False)
net = net.module

criterion = {
    'mse': nn.MSELoss().cuda(),
    'l1': nn.L1Loss().cuda()
}

label_list = []
pred_list = []

if not os.path.exists('output/'):
        os.makedirs('output/')

patches = []
for i, data in tqdm(enumerate(dataloader)):
    data = data.cuda()
    loss, pred, mask = net(data, block=True)
    loss = torch.mean(loss)
    save_path = 'output/' + config.comment + '_' + str(i) + '_' + str(round(loss.item(), 4)) + '_real.png'

    output = net.unpatchify(pred)

    mask = mask.unsqueeze(2)
    masked_patchified_data = net.patchify(data) * (1 - mask)
    masked_data = net.unpatchify(masked_patchified_data)
    merge = net.unpatchify(masked_patchified_data + pred * mask)
    diff = torch.abs(merge - data) * 4
    save_image(torch.cat([data, masked_data, merge, diff]), save_path, nrow=4)

    print(i, 'Loss: ', loss.item())

