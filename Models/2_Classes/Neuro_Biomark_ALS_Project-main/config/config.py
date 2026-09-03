import torch
import os

class Config():
    dirname = os.path.dirname(__file__) #On config level
    project_dir = os.path.join(dirname, '..') #On project main level
    dataset_dir_path = project_dir + r"/dataset"
    logs_dir_path = project_dir + r"/logs"
    saved_models_dir_path = project_dir + r"/saved_models"
    no_of_folds = 5
    fold_no = -1

    lr = 1e-4
    batch_size = 16
    no_of_epoch = 50
    weight_decay = 0.01
    early_stop = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing {device} device")

config = Config()