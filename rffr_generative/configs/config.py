class DefaultConfigs(object):
    # actual parameters
    seed = 912
    comment = 'RFFR: Generative modeling.'
    weight_decay = 0.05
    beta1 = 0.9     # 0.9 for MAE
    beta2 = 0.95    # 0.95 for MAE
    model = 'mae'

    batch_size = 600 # batch size: n * 2
    lr = 1.5e-4 * batch_size / 256
    lr_warmup = 2500
    max_iter = 25000        # To be modified according to data size
    gpus = '0, 1'
    
    iter_per_epoch = 500
    dataset_name = ['Faceforensics'] # dataset need to be tested

    # paths information
    my_pretrained = None
    in_pretrained = None
    checkpoint_path = './checkpoint/' + model + '/current_model/'
    best_model_path = './checkpoint/' + model + '/CDF/'
    enable_tensorboard = True
    logs = './logs/'

    lq = False
    aug = False

    real_label_path = 'FN/train/real_train_label.json'
    val_label_path = 'FN/train/real_train_label.json'

    save_code = ['configs/config.py', 'train.py', \
                'models/model_mae.py', 'dataset.py', 'utils.py']
    
    
config = DefaultConfigs()
