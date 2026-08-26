import torch

class Config():

    dataset_dir_path = r"C:/Users/ab4197/Documents/NeuroBiomark_Project/Models/Model_with_training_no_control_class/Neuro_Biomark_ALS_Project-main/dataset"
    logs_dir_path = r"C:/Users/ab4197/Documents/NeuroBiomark_Project/Models/Model_with_training_no_control_class/Neuro_Biomark_ALS_Project-main/logs"
    saved_models_dir_path = r"C:/Users/ab4197/Documents/NeuroBiomark_Project/Models/Model_with_training_no_control_class/Neuro_Biomark_ALS_Project-main/saved_models"

    no_of_folds = 5
    fold_no = -1

    lr = 1e-4
    batch_size = 16
    no_of_epoch = 50
    weight_decay = 0.01
    early_stop = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing {device} device")

config = Config()