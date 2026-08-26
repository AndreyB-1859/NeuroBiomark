import os

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
import re
from config.config import config
from src.process_data.utils import get_train_test, compute_rgb_mean_std, transform_function, ALSDataset, augment_images, compute_class_weights


from src.models.DenseNet121 import AttentionDenseNet121

from src.models.augmentation_evaluation_CNN import plot_results
from src.models.augmentation_evaluation_CNN import plot_accuracy_comparison
import json

results_df = pd.read_csv('Models/Model_with_training_no_control_class/Neuro_Biomark_ALS_Project-main/logs/augmentation_evaluation/basiccnn_20260825_171153/augmentation_comparison.csv')

save_dir = 'Models/Model_with_training_no_control_class/Neuro_Biomark_ALS_Project-main/test/'

with open('Models/Model_with_training_no_control_class/Neuro_Biomark_ALS_Project-main/logs/augmentation_evaluation/basiccnn_20260825_171153/best_fold_logs.json') as f:
    best_folds = json.load(f)

plot_results(results_df, best_folds, save_dir)





