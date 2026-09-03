"""
Data Loading Integration for BasicCNN Augmentation Evaluation

This module integrates with your existing data pipeline (prepare_group_folds)
and creates dataloaders with proper augmentation handling.

NORMALIZATION STRATEGY:
Since we're training from scratch (not using pretrained weights), we use
dataset-specific normalization computed from your 190 ALS images rather than
ImageNet statistics. This gives better results for medical imaging.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.process_data.utils import get_train_test, compute_class_weights


# ============================================================================
# NORMALIZATION COMPUTATION
# ============================================================================

def compute_dataset_statistics(image_paths, sample_size=None):
    """
    Compute mean and std from dataset images
    
    Args:
        image_paths: List of image paths
        sample_size: If set, only use this many images (for speed)
    
    Returns:
        mean, std as numpy arrays [R, G, B]
    """
    print("\n" + "="*70)
    print("Computing dataset-specific normalization statistics...")
    print("="*70)
    
    if sample_size and sample_size < len(image_paths):
        # Sample random images for faster computation
        np.random.seed(42)
        indices = np.random.choice(len(image_paths), sample_size, replace=False)
        sample_paths = [image_paths[i] for i in indices]
        print(f"Using {sample_size} random images for statistics computation")
    else:
        sample_paths = image_paths
        print(f"Using all {len(image_paths)} images")
    
    # Accumulate pixel values
    pixel_sum = np.zeros(3)
    pixel_sum_sq = np.zeros(3)
    total_pixels = 0
    
    for img_path in sample_paths:
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Accumulate statistics
        pixel_sum += img.reshape(-1, 3).sum(axis=0)
        pixel_sum_sq += (img.reshape(-1, 3) ** 2).sum(axis=0)
        total_pixels += img.shape[0] * img.shape[1]
    
    # Compute mean and std
    mean = pixel_sum / total_pixels
    std = np.sqrt((pixel_sum_sq / total_pixels) - (mean ** 2))
    
    print(f"\nDataset Statistics:")
    print(f"  Mean: R={mean[0]:.4f}, G={mean[1]:.4f}, B={mean[2]:.4f}")
    print(f"  Std:  R={std[0]:.4f}, G={std[1]:.4f}, B={std[2]:.4f}")
    print(f"\nCompare to ImageNet:")
    print(f"  ImageNet Mean: R=0.485, G=0.456, B=0.406")
    print(f"  ImageNet Std:  R=0.229, G=0.224, B=0.225")
    print("="*70 + "\n")
    
    return mean, std




def get_normalization_stats(train_paths):
    """
    Get or compute dataset normalization statistics
    
    Args:
        train_paths: Training image paths
        recompute: Force recomputation even if cached
    
    Returns:
        mean, std as numpy arrays
    """
    return compute_dataset_statistics(train_paths)


# ============================================================================
# CUSTOM DATASET
# ============================================================================

class ALSDatasetWithAugmentation(Dataset):
    """
    Dataset for ALS classification with augmentation support
    
    Handles:
    - RGB image loading
    - Albumentations transforms
    - OpenCV-based augmentations
    - Dataset-specific normalization (NOT ImageNet)
    """
    
    def __init__(self, image_paths, labels, transform=None, aug_name='none', 
                 mean=None, std=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (0, 1, 2)
            transform: Albumentations transform or None
            aug_name: Name of augmentation (for OpenCV augs)
            mean: Dataset mean [R, G, B] (if None, uses default)
            std: Dataset std [R, G, B] (if None, uses default)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.aug_name = aug_name
        
        # Use provided stats or default to ImageNet (fallback)
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
            self.using_custom_norm = True
        else:
            # Fallback to ImageNet if stats not provided
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            self.using_custom_norm = False
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)  # Convert to numpy array
        
        # Apply augmentation
        if self.transform is not None:
            if self.aug_name.startswith('opencv_'):
                # Apply OpenCV augmentation first
                img = self._apply_opencv_augmentation(img, self.aug_name)
                # Resize to target size (400x400)
                img = cv2.resize(img, (400, 400))
                # Convert to tensor
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                # Normalize with dataset stats
                img = self._normalize_tensor(img)
            else:
                # Apply Albumentations transform (includes normalization)
                augmented = self.transform(image=img)
                img = augmented['image']
        else:
            # No augmentation, just resize and normalize
            img = cv2.resize(img, (400, 400))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = self._normalize_tensor(img)
        
        label = self.labels[idx]
        
        return img, label
    
    def _normalize_tensor(self, img_tensor):
        """Apply dataset-specific normalization"""
        return (img_tensor - self.mean) / self.std
    
    def _apply_opencv_augmentation(self, img, aug_name):
        """Apply OpenCV-based augmentations"""
        h, w = img.shape[:2]
        
        if aug_name == 'opencv_flip':
            # Random horizontal + vertical flip
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)  # Horizontal
            if np.random.rand() < 0.3:
                img = cv2.flip(img, 0)  # Vertical
        
        elif aug_name == 'opencv_rotate_90':
            # Random 90-degree rotation
            k = np.random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
            if k > 0:
                img = cv2.rotate(img, [None, cv2.ROTATE_90_CLOCKWISE, 
                                      cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][k])
        
        elif aug_name == 'opencv_grayscale':
            # Convert to grayscale and back to 3 channels
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        elif aug_name == 'opencv_gray_denoise':
            # Grayscale + denoising
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            img = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        
        elif aug_name == 'opencv_color_denoise':
            # Color denoising
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        elif aug_name == 'opencv_scale':
            # Random scaling
            scale = np.random.uniform(0.8, 1.2)
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(img, (new_w, new_h))
            # Pad or crop to original size
            if scale < 1.0:
                pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                img = cv2.copyMakeBorder(scaled, pad_h, h - new_h - pad_h, 
                                       pad_w, w - new_w - pad_w, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
                img = scaled[start_h:start_h+h, start_w:start_w+w]
        
        elif aug_name == 'opencv_crop':
            # Random crop
            crop_size = 10
            if h > 2 * crop_size and w > 2 * crop_size:
                cropped = img[crop_size:(h-crop_size), crop_size:(w-crop_size)]
                img = cv2.resize(cropped, (w, h))
        
        elif aug_name == 'opencv_noise':
            # Add random noise
            noise = np.random.randint(0, 10, img.shape, dtype="uint8")
            img = cv2.add(img, noise)
        
        elif aug_name == 'opencv_perspective':
            # Perspective transformation
            src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
            dst_points = src_points + np.random.normal(0, 5, src_points.shape).astype(np.float32)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            img = cv2.warpPerspective(img, M, (w, h))
        
        return img


# ============================================================================
# AUGMENTATION TRANSFORMS
# ============================================================================

def get_augmentation_transform(aug_name, image_size=400, mean=None, std=None):
    """
    Get Albumentations transform for specified augmentation
    
    Args:
        aug_name: Name of augmentation technique
        image_size: Target image size (default: 400)
        mean: Dataset mean [R, G, B] for normalization
        std: Dataset std [R, G, B] for normalization
    
    Returns:
        Albumentations Compose object
    """
    
    # Use provided stats or fallback to ImageNet
    if mean is not None and std is not None:
        norm_mean = mean.tolist() if hasattr(mean, 'tolist') else list(mean)
        norm_std = std.tolist() if hasattr(std, 'tolist') else list(std)
    else:
        # Fallback to ImageNet (though dataset-specific is preferred)
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    
    # Base transforms
    base_transform = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2()
    ]
    
    # Augmentation-specific transforms
    aug_dict = {
        'none': [],
        
        # Geometric
        'horizontal_flip': [A.HorizontalFlip(p=0.5)],
        'vertical_flip': [A.VerticalFlip(p=0.5)],
        'random_rotation': [A.Rotate(limit=20, p=0.7)],
        'shift_scale_rotate': [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                             rotate_limit=15, p=0.7)
        ],
        'random_zoom': [A.RandomScale(scale_limit=0.2, p=0.5)],
        'elastic_deformation': [A.ElasticTransform(alpha=1, sigma=50, p=0.3)],
        
        # Color/Intensity
        'brightness_contrast': [
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                      contrast_limit=0.2, p=0.7)
        ],
        'color_jitter': [
            A.ColorJitter(brightness=0.2, contrast=0.2, 
                         saturation=0.2, hue=0.1, p=0.7)
        ],
        'clahe': [A.CLAHE(clip_limit=4.0, p=0.5)],
        'grayscale': [A.ToGray(p=0.3)],
        
        # Noise and Blur
        'gaussian_noise': [A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)],
        'gaussian_blur': [A.GaussianBlur(blur_limit=(3, 7), p=0.5)],
        
        # OpenCV-based (placeholder - handled in dataset)
        'opencv_flip': [],
        'opencv_rotate_90': [],
        'opencv_grayscale': [],
        'opencv_gray_denoise': [],
        'opencv_color_denoise': [],
        'opencv_scale': [],
        'opencv_crop': [],
        'opencv_noise': [],
        'opencv_perspective': [],
    }
    
    # Get augmentation
    aug_transforms = aug_dict.get(aug_name, [])
    
    # Combine
    transform = A.Compose(aug_transforms + base_transform)
    
    return transform


# ============================================================================
# DATALOADER CREATION
# ============================================================================

def create_dataloaders_for_fold(fold_num, aug_name, batch_size=16, num_workers=4):
    """
    Create train and validation dataloaders for a specific fold and augmentation
    
    Args:
        fold_num: Fold number (0-4 for 5-fold CV)
        aug_name: Name of augmentation technique
        batch_size: Batch size (default: 8)
        num_workers: Number of workers for dataloader
    
    Returns:
        train_loader, val_loader, class_weights, (mean, std)
    """
    
    # Get train and test data for this fold
    # Your existing function returns: train_paths, val_paths, train_labels, val_labels
    train_paths, val_paths, train_labels, val_labels = get_train_test(fold_num)
    
    # Convert to numpy arrays if needed
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    
    print(f"\nFold {fold_num} - Augmentation: {aug_name}")
    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    print(f"Train class distribution: {np.bincount(train_labels)}")
    print(f"Val class distribution: {np.bincount(val_labels)}")
    
    # Compute dataset-specific normalization from training data
    # This is computed once per fold and reused for all augmentations
    mean, std = get_normalization_stats(train_paths)
    
    # Get augmentation transform with dataset-specific normalization
    train_transform = get_augmentation_transform(aug_name, image_size=400, 
                                                 mean=mean, std=std)
    val_transform = get_augmentation_transform('none', image_size=400, 
                                               mean=mean, std=std)
    
    # Create datasets
    train_dataset = ALSDatasetWithAugmentation(
        train_paths, train_labels, train_transform, aug_name, mean, std
    )
    val_dataset = ALSDatasetWithAugmentation(
        val_paths, val_labels, val_transform, 'none', mean, std
    )
    
    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(train_labels)
    class_weights = torch.FloatTensor(class_weights)
    
    print(f"Class weights: {class_weights.numpy()}")
    print(f"Using dataset-specific normalization")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, class_weights, (mean, std)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing data loading integration...")
    
    # Test creating dataloaders for fold 0 with no augmentation
    try:
        # First, make sure prepare_group_folds has been called
        from src.process_data.load_data import prepare_group_folds
        print("Preparing group folds...")
        prepare_group_folds()
        
        train_loader, val_loader, class_weights, (mean, std) = create_dataloaders_for_fold(
            fold_num=0,
            aug_name='none',
            batch_size=8,
            num_workers=0  # Use 0 for testing
        )
        
        print("\n✅ Dataloader creation successful!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Class weights: {class_weights}")
        print(f"Dataset normalization:")
        print(f"  Mean: {mean}")
        print(f"  Std:  {std}")
        
        # Test loading one batch
        images, labels = next(iter(train_loader))
        print(f"\nBatch test:")
        print(f"Images shape: {images.shape}")  # Should be [8, 3, 400, 400]
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Labels: {labels.numpy()}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()