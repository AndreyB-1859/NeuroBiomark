import os
import sys
import shutil
from PIL import Image
from collections import Counter
import csv

import pandas as pd
import cv2
import numpy as np
import torch
from torch.nn.functional import one_hot
from torchvision import transforms
from torch.utils.data import Dataset

from config.config import config

def get_train_test(fold):

    print(f"\tsplitting into train and test for fold: {fold}")
    # Loading the image key's with fold file
    keys_path = os.path.join(config.dataset_dir_path, "image_keys_with_fold")
    df = pd.read_csv(keys_path)

    # Label mapping
    label_map = {
        "Concordant": 0,
        "Discordant": 1
    }

    # Splitting based on fold
    train_df = df[df["fold"] != fold]
    val_df = df[df["fold"] == fold]

    # Group-level non-overlap check
    train_groups = set(train_df["GroupID"])
    val_groups = set(val_df["GroupID"])

    overlap = train_groups.intersection(val_groups)
    if overlap:
        print(f"WARNING: {len(overlap)} overlapping patient(s): {overlap}")
    else:
        print("No overlapping patients between train and validation folds")

    # Print and  save patient IDs per fold
    print(f"\nUnique patients in train ({len(train_groups)}):")
    print(sorted(train_groups))
    print(f"\nUnique patients in val ({len(val_groups)}):")
    print(sorted(val_groups))

    # Create DataFrames mapping GroupID → Category
    train_groups_df = (
        train_df.groupby("GroupID")["Category"]
        .first()  # since each GroupID has same Category
        .reset_index()
)
    val_groups_df = (
        val_df.groupby("GroupID")["Category"]
        .first()
        .reset_index()
    )

    # Saving them to CSV for traceability
    save_dir = os.path.join(config.dataset_dir_path, "fold_patient_lists")
    os.makedirs(save_dir, exist_ok=True)
    train_groups_df.to_csv(os.path.join(save_dir, f"fold_{fold}_train_patients.csv"), index=False)
    val_groups_df.to_csv(os.path.join(save_dir, f"fold_{fold}_val_patients.csv"), index=False)


    # Creating image paths and labels
    def make_paths_and_labels(sub_df):
        image_paths = [os.path.join(config.dataset_dir_path,"images",f"{int(row['Image No'])}.tif") for _,row in sub_df.iterrows()]
        labels = [label_map[row["Category"]] for _,row in sub_df.iterrows()]
        return image_paths, labels

    train_image_paths, train_labels = make_paths_and_labels(train_df)
    val_image_paths, val_labels = make_paths_and_labels(val_df)

    #printing data distribution stats
    print(f"\ttrain data distribution:- {Counter(train_labels)}")
    print(f"\tval data distribution:- {Counter(val_labels)}")

    return train_image_paths, val_image_paths, train_labels, val_labels

def compute_rgb_mean_std(image_paths):

    print("\tcalculating the mean and std of original training data to normalize the images")

    # transform fun to convert images to tensors
    to_tensor = transforms.ToTensor()

    # accumulators
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_pixels = 0

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = to_tensor(img)

        channel_sum += img_tensor.sum(dim=(1,2))
        channel_squared_sum += (img_tensor**2).sum(dim=(1,2))
        num_pixels += img_tensor.shape[1]* img_tensor.shape[2]

    mean = channel_sum / num_pixels
    std = (channel_squared_sum / num_pixels - mean**2).sqrt()

    return mean.tolist(), std.tolist()

def transform_function(mean, std):
    return transforms.Compose([
        transforms.Resize((400,400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])

class ALSDataset(Dataset):

    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def save_augmented_image(folder, base_name, suffix, image, train_image_paths, train_labels, label):
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"{base_name}_{suffix}.tif")
    cv2.imwrite(out_path, image)
    train_image_paths.append(out_path)
    train_labels.append(label)

def augment_images(img_paths, labels):

    data_folder = os.path.join(config.dataset_dir_path, 'augmented_training_data')

    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
        print(f"\tDeleted folder: {data_folder}")
    else:
        print(f"\tFolder does not exist: {data_folder}")

    print("\taugmenting training data")

    train_image_paths = []
    train_labels = []

    for img_path, label in zip(img_paths, labels):
        # the original image
        print(img_path)
        img_color = cv2.imread(img_path)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        original_folder = os.path.join(data_folder, "original")
        if not os.path.exists(original_folder):
            os.makedirs(original_folder)
        output_path = os.path.join(original_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, img_color)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # flipping the image
        flipped_img = cv2.flip(img_color, 1)

        flip_folder = os.path.join(data_folder, "flipped")
        if not os.path.exists(flip_folder):
            os.makedirs(flip_folder)
        output_path = os.path.join(flip_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, flipped_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # rotating the image
        h, w = img_color.shape[:2]
        center = (w //2, h//2)
        m= cv2.getRotationMatrix2D(center, 90, 1)
        rotated_img = cv2.warpAffine(img_color, m, (h, w))

        rotate_folder = os.path.join(data_folder, "rotated")
        if not os.path.exists(rotate_folder):
            os.makedirs(rotate_folder)
        output_path = os.path.join(rotate_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, rotated_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # converting into grayscale (but retains 3-channels)
        grayscale_3channel_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        gray_folder = os.path.join(data_folder, "gray")
        if not os.path.exists(gray_folder):
            os.makedirs(gray_folder)
        output_path = os.path.join(gray_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, grayscale_3channel_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # edge detection of images
        edges = cv2.Canny(img_gray, 50, 150)
        dilated_edges = cv2.dilate(edges, None, iterations=2)
        mask = cv2.threshold(dilated_edges, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # grayscale denoising of images
        gray_denoised_img = cv2.bilateralFilter(img_gray, d=15, sigmaColor=30, sigmaSpace=75)
        gray_denoised_img = cv2.bitwise_and(gray_denoised_img, mask) + cv2.bitwise_and(img_gray, cv2.bitwise_not(mask))
        gray_3channel_denoised_img = cv2.cvtColor(gray_denoised_img, cv2.COLOR_GRAY2BGR)

        gray_denoising = os.path.join(data_folder, "gray_denoising")
        if not os.path.exists(gray_denoising):
            os.makedirs(gray_denoising)
        output_path = os.path.join(gray_denoising, os.path.basename(img_path))
        cv2.imwrite(output_path, gray_3channel_denoised_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # color denoising of images
        color_denoised_img = cv2.bilateralFilter(img_color, d=15, sigmaColor=30, sigmaSpace=75)
        color_denoised_img = cv2.bitwise_and(color_denoised_img, color_denoised_img,mask=mask) + cv2.bitwise_and(img_color, img_color,mask=cv2.bitwise_not(mask))

        color_denoising = os.path.join(data_folder, "color_denoising")
        if not os.path.exists(color_denoising):
            os.makedirs(color_denoising)
        output_path = os.path.join(color_denoising, os.path.basename(img_path))
        cv2.imwrite(output_path, color_denoised_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # scaling the image to scale FACTOR 0.9
        scaled_img = cv2.resize(img_color, (int(h*0.9), int(w*0.9)))

        scale_folder = os.path.join(data_folder, "scaled")
        if not os.path.exists(scale_folder):
            os.makedirs(scale_folder)
        output_path = os.path.join(scale_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, scaled_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # cropping the image with crop size (10,10)
        cropped_img = img_color[10:(h-10), 10:(w-10)]

        crop_folder = os.path.join(data_folder, "cropped")
        if not os.path.exists(crop_folder):
            os.makedirs(crop_folder)
        output_path = os.path.join(crop_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, cropped_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # adding noise to image at noice level 10
        noise = np.random.randint(0,10,img_color.shape, dtype="uint8")
        noisy_img =cv2.add(img_color, noise)

        noise_folder = os.path.join(data_folder, "noisy")
        if not os.path.exists(noise_folder):
            os.makedirs(noise_folder)
        output_path = os.path.join(noise_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, noisy_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # perspective transformation of images
        src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        dst_points = src_points + np.random.normal(0, 5, src_points.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        trans_img = cv2.warpPerspective(img_color, M, (w, h))

        perspective_folder = os.path.join(data_folder, "perspective_transformed")
        if not os.path.exists(perspective_folder):
            os.makedirs(perspective_folder)
        output_path = os.path.join(perspective_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, trans_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

    return train_image_paths, train_labels


def compute_class_weights(labels, num_classes=2):
    counter = Counter(labels)
    total = sum(counter.values())

    class_weights = []
    for i in range(num_classes):
        class_count = counter.get(i,0)
        if class_count > 0:
            weight = total/(num_classes*class_count)
        else:
            weight = 0
        class_weights.append(weight)

    return torch.tensor(class_weights, dtype=torch.float)