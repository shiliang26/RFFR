from .label_path import get_label_path

class DefaultConfigs(object):

    # Setting
    seed = 42
    comment = 'Open-sourced code.'

    # Models
    lr = 2e-5
    batch_size = 16 # total batch size: batch size * 2
    gpus = '4'
    model = 'rffr'
    mae_path = '../pretrain/MAE/FF_FN_0.00147_10000.pth.tar'
    pretrained_weights = '../pretrain/jx_vit_base_p16_224-80ecf9dd.pth'

    # Data
    protocol = 'F2F_All'
    dataset_base = '/data/shiliang/data/'
    real_label_path, fake_label_path, val_label_path, test_label_path, metrics = get_label_path(protocol)
    
    # Schedule
    max_iter = 500000
    iter_per_epoch = 50
    iter_per_eval = 50

    # paths information
    checkpoint_path = './checkpoint/' + model + '/current_model/'
    best_model_path = './checkpoint/' + model + '/best_model/'
    logs = './logs/'
    
    # Code Saver
    save_code = ['configs/config.py', 'train.py', \
                 'utils/simple_evaluate.py', 'utils/dataset.py', \
                 'models/model_mae.py', 'models/model_detector.py']
    
config = DefaultConfigs()
