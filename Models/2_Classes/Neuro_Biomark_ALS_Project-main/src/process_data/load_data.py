import os

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
import re
from torch.utils.data import DataLoader
from config.config import config
from src.process_data.utils import get_train_test, compute_rgb_mean_std, transform_function, ALSDataset, augment_images, compute_class_weights

def create_groups(df):
    """
    Standardize and extract group IDs from Case ID column.

    Example:
        'SD12 Discord BA44'  -> 'SD012'
        'SD012-13 Discord BA44' -> 'SD012-13'

    Returns:
        df with new column 'GroupID'
    """
    def extract_group(case_id):
        case_id = str(case_id).strip()
        # Match patterns like SD12, SD012, SD12-13, SD012-13
        match = re.search(r"(SD\d{1,3}(?:-\d{1,3})?)", case_id)
        if not match:
            return None

        group = match.group(1)

        # Split by dash if it exists
        if "-" in group:
            main, sub = group.split("-")
            main_num = re.search(r"\d+", main).group()
            # Zero-pad main number to 3 digits
            main_padded = f"SD{int(main_num):03d}"
            group = f"{main_padded}-{sub}"
        else:
            main_num = re.search(r"\d+", group).group()
            group = f"SD{int(main_num):03d}"

        return group

    # Apply extraction
    df["GroupID"] = df["Case ID"].apply(extract_group)

    # Report issues
    missing = df["GroupID"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} Case IDs could not be parsed properly!")

    return df

def prepare_leave_one_group_out_folds():
    """
    Prepare leave-one-group-out folds, ensuring all samples from the same patient
    stay together in one fold.
    """
    # Load the image key's excel file
    keys_path = os.path.join(config.dataset_dir_path, "image_keys.xlsx")
    df = pd.read_excel(keys_path, sheet_name="Sheet1", header=1)
    df = df[["Image No", "Case ID", "Category"]].copy()

    # Clean and extract group IDs
    df = create_groups(df)

    # Initialize Leave-One-Group-Out CV
    logo = LeaveOneGroupOut()

    # Create a column for fold assignments
    df["fold"] = -1

    # Apply the split
    for fold, (train_idx, val_idx) in enumerate(
        logo.split(X=df, y=df["Category"], groups=df["GroupID"])
    ):
        df.loc[val_idx, "fold"] = fold

    n_folds = df["fold"].nunique()

    # Save the new DataFrame
    group_key_path = os.path.join(config.dataset_dir_path, "image_keys_with_fold")
    df.to_csv(group_key_path, index=False)

    # print(f"Saved Leave-One-Group-Out folds to: {group_key_path}")
    # print(f"Total folds created = {df['fold'].nunique()} (equal to # unique patients)")
    return n_folds


def prepare_group_folds():
    # Load the image key's excel file
    keys_path = os.path.join(config.dataset_dir_path, "image_keys.xlsx")
    df = pd.read_excel(keys_path, sheet_name="Sheet1", header=1)
    # Remove the control class from the dataset
    for i, row in df.iterrows():
        if row['Condition'] == 'Control':
            df = df.drop(index=i)

    df = df.reset_index()


    df = df[["Image No", "Case ID", "Category"]].copy()

    # Clean and extract group IDs
    df = create_groups(df)

    # Initialize StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(
        n_splits=config.no_of_folds,
        shuffle=True,
        random_state=42
    )

    # Create a column for fold assignments
    df["fold"] = -1

    # Apply the split
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(X=df, y=df["Category"], groups=df["GroupID"])
    ):
        df.loc[val_idx, "fold"] = fold

    # Save the new DataFrame
    group_key_path = os.path.join(config.dataset_dir_path, "image_keys_with_fold")
    df.to_csv(group_key_path, index=False)
    # print(f"Saved grouped folds to: {group_key_path}")


def prepare_folds():

    # Loading the image key's excel file
    keys_path = os.path.join(config.dataset_dir_path, "image_keys.xlsx")
    df = pd.read_excel(keys_path, sheet_name="Sheet1", header=1)
    df = df[["Image No", "Case ID", "Category"]].copy()

    # Initializing StratifiedGroupKFold
    sgkf = StratifiedKFold(n_splits=config.no_of_folds, shuffle=True, random_state=42)

    # Creating a new colum in the DataFrame to store fold assignments
    df["fold"] = -1

    # Applying the split and assign folds
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X=df, y=df["Category"])):
        df.loc[val_idx, "fold"] = fold

    # Saving the DataFrame
    keys_path = os.path.join(config.dataset_dir_path,f"image_keys_with_fold")
    df.to_csv(keys_path, index=False)


def get_dataloaders_and_classweights(fold):

    # getting the train and test dats for the fold
    train_image_paths, val_image_paths, train_labels, val_labels = get_train_test(fold)

    # function to normalize the images before passing into model
    mean, std = compute_rgb_mean_std(train_image_paths)
    transform =  transform_function(mean,std)

    # augmenting the train images
    train_image_paths, train_labels = augment_images(train_image_paths, train_labels)

    # getting the class_weights for loss calculation
    class_weights = compute_class_weights(train_labels)


    # instantiating the torch datasets
    train_dataset = ALSDataset(train_image_paths, train_labels, transform)
    val_dataset = ALSDataset(val_image_paths, val_labels, transform)

    print("\tinstantiating the train and val dataloaders")
    # instantiating the torch dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, class_weights
