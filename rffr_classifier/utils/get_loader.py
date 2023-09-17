from configs.config import config
import json
from torch.utils.data import DataLoader
from utils.dataset import Deepfake_Dataset


def get_dataset():
    print('\nLoad Train Data')

    train_real_dict = []
    for real_dataset in config.real_label_path:
        with open('../data_label/' + real_dataset) as f:
            train_real_dict += json.load(f)
    print("Train data frames (real): ", len(train_real_dict))

    train_fake_dict = []
    for fake_dataset in config.fake_label_path:
        with open('../data_label/' + fake_dataset) as f:
            train_fake_dict += json.load(f)
    print("Train data frames (fake): ", len(train_fake_dict))

    print('\nLoad Test Data')
    if isinstance(config.val_label_path, str):
        test_json = open('../data_label/' + config.val_label_path)
        test_dict = json.load(test_json)
        test_dataloader = DataLoader(Deepfake_Dataset(test_dict, train=False), batch_size=config.batch_size, shuffle=False)
    else:
        test_dataloader = []
        for path in config.val_label_path:
            test_json = open('../data_label/' + path)
            test_dict = json.load(test_json)
            test_dataloader.append(DataLoader(Deepfake_Dataset(test_dict, train=False), batch_size=16, shuffle=False))


    train_real_dataloader = DataLoader(Deepfake_Dataset(train_real_dict, train=True), batch_size=config.batch_size, shuffle=True)
    train_fake_dataloader = DataLoader(Deepfake_Dataset(train_fake_dict, train=True), batch_size=config.batch_size, shuffle=True)

    return train_real_dataloader, train_fake_dataloader, test_dataloader