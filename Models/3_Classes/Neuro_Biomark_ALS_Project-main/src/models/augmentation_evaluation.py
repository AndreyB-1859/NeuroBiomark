# ============================================================================
# src/models/augmentation_evaluation.py - CORRECTED VERSION
# ============================================================================
# Evaluates different augmentation techniques to find the best ones
# ============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, precision_recall_fscore_support
from collections import Counter
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2  # Added for OpenCV-based augmentations

# Import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: albumentations not installed. Install with: pip install albumentations")
    ALBUMENTATIONS_AVAILABLE = False

from config.config import config
from src.process_data.utils import get_train_test, compute_class_weights, compute_rgb_mean_std
from src.models.EfficientnetB0 import EfficientNetB0Classifier


# ============================================================================
# 1. AUGMENTATION LIBRARY
# ============================================================================

class AugmentationLibrary:
    """Library of augmentation techniques using Albumentations"""

    def apply_clahe_to_lab(image):
        """Apply CLAHE only to L channel in LAB color space"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    @staticmethod
    def get_augmentation_transform(aug_name, img_size=400):
        """Returns Albumentations transform for a given augmentation"""
        
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations not installed")
        
        base = [A.Resize(img_size, img_size)]
        to_tensor = [ToTensorV2()]
        
        augmentations = {
            # Baseline
            'none': A.Compose(base + to_tensor),
            
            # Tier 1: Must Have
            'horizontal_flip': A.Compose(base + [
                A.HorizontalFlip(p=1.0),
            ] + to_tensor),
            
            'vertical_flip': A.Compose(base + [
                A.VerticalFlip(p=1.0),
            ] + to_tensor),
            
            'random_rotation': A.Compose(base + [
                A.Rotate(limit=15, p=1.0, border_mode=0),
            ] + to_tensor),
            
            'elastic_deformation': A.Compose(base + [
                A.ElasticTransform(alpha=30, sigma=30*0.05, alpha_affine=30*0.03, p=0.5),
            ] + to_tensor),
            
            'brightness_contrast': A.Compose(base + [
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ] + to_tensor),
            
            'random_zoom': A.Compose(base + [
                A.RandomScale(scale_limit=0.1,p=1.0),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
                A.CenterCrop(img_size, img_size),
            ] + to_tensor),
            
            # Tier 2: Highly Recommended
            'clahe': A.Compose(base + [
                A.CLAHE(clip_limit=2.0,tile_grid_size=(8, 8), p=1.0),
            ] + to_tensor),
            
            'gaussian_noise': A.Compose(base + [
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ] + to_tensor),
            
            'gaussian_blur': A.Compose(base + [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ] + to_tensor),
            
            'color_jitter': A.Compose(base + [
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, 
                         p=1.0),
            ] + to_tensor),
            
            # Tier 3: Optional
            'shift_scale_rotate': A.Compose(base + [
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10,p=0.3,border_mode=0),
            ] + to_tensor),
            
            # Your Original (for comparison)
            'grayscale': A.Compose(base + [
                A.ToGray(p=1.0),
                A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1) if x.ndim == 2 else x),
            ] + to_tensor),
        }
        
        return augmentations.get(aug_name, augmentations['none'])
    
    @staticmethod
    def get_all_augmentation_names():
        """Returns list of all augmentation names to test"""
        return [
            'none',  # Baseline
            'horizontal_flip', 'vertical_flip', 'random_rotation',
            'elastic_deformation', 'brightness_contrast', 'random_zoom',
            'clahe', 'gaussian_noise', 'gaussian_blur', 'color_jitter',
            'shift_scale_rotate', 'grayscale',
            # OpenCV-based standard augmentations (for comparison)
            'opencv_flip', 'opencv_rotate_90', 'opencv_grayscale', 
            'opencv_gray_denoise', 'opencv_color_denoise',
            'opencv_scale', 'opencv_crop', 'opencv_noise', 'opencv_perspective'
        ]


    
    @staticmethod
    def apply_opencv_augmentation(image, aug_name):
        """
        Apply OpenCV-based augmentation to a numpy image array (RGB format)
        These are the standard augmentations from your original code
        Returns: augmented image in RGB format
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        if aug_name == 'opencv_flip':
            # Horizontal flip
            return cv2.flip(img, 1)
        
        elif aug_name == 'opencv_rotate_90':
            # Rotate 90 degrees
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 90, 1)
            return cv2.warpAffine(img, M, (h, w))
        
        elif aug_name == 'opencv_grayscale':
            # Convert to grayscale but keep 3 channels
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        elif aug_name == 'opencv_gray_denoise':
            # Grayscale denoising with edge detection
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, 50, 150)
            dilated_edges = cv2.dilate(edges, None, iterations=2)
            mask = cv2.threshold(dilated_edges, 127, 255, cv2.THRESH_BINARY_INV)[1]
            gray_denoised = cv2.bilateralFilter(img_gray, d=15, sigmaColor=30, sigmaSpace=75)
            gray_denoised = cv2.bitwise_and(gray_denoised, mask) + cv2.bitwise_and(img_gray, cv2.bitwise_not(mask))
            return cv2.cvtColor(gray_denoised, cv2.COLOR_GRAY2RGB)
        
        elif aug_name == 'opencv_color_denoise':
            # Color denoising with edge detection
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, 50, 150)
            dilated_edges = cv2.dilate(edges, None, iterations=2)
            mask = cv2.threshold(dilated_edges, 127, 255, cv2.THRESH_BINARY_INV)[1]
            color_denoised = cv2.bilateralFilter(img, d=15, sigmaColor=30, sigmaSpace=75)
            color_denoised = cv2.bitwise_and(color_denoised, color_denoised, mask=mask) + \
                            cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
            return color_denoised
        
        elif aug_name == 'opencv_scale':
            # Scale to 0.9x
            scaled = cv2.resize(img, (int(w * 0.9), int(h * 0.9)))
            # Pad back to original size
            pad_h = (h - scaled.shape[0]) // 2
            pad_w = (w - scaled.shape[1]) // 2
            return cv2.copyMakeBorder(scaled, pad_h, h - scaled.shape[0] - pad_h, 
                                     pad_w, w - scaled.shape[1] - pad_w, 
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        elif aug_name == 'opencv_crop':
            # Crop 10 pixels from each side
            crop_size = 10
            if h > 2 * crop_size and w > 2 * crop_size:
                cropped = img[crop_size:(h - crop_size), crop_size:(w - crop_size)]
                # Resize back to original size
                return cv2.resize(cropped, (w, h))
            return img
        
        elif aug_name == 'opencv_noise':
            # Add random noise (level 10)
            noise = np.random.randint(0, 10, img.shape, dtype="uint8")
            return cv2.add(img, noise)
        
        elif aug_name == 'opencv_perspective':
            # Perspective transformation
            src_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
            dst_points = src_points + np.random.normal(0, 5, src_points.shape).astype(np.float32)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            return cv2.warpPerspective(img, M, (w, h))
        
        else:
            return img


# ============================================================================
# 2. CUSTOM DATASET WITH AUGMENTATION
# ============================================================================

class AugmentedALSDataset(torch.utils.data.Dataset):
    """Dataset that applies a single augmentation technique"""
    
    def __init__(self, image_paths, labels, aug_transform, aug_name='none'):
        self.image_paths = image_paths
        self.labels = labels
        self.aug_transform = aug_transform
        # self.mean = mean
        # self.std = std
        self.aug_name = aug_name  # Store augmentation name to identify OpenCV augs
        
        # Standard ImageNet normalization for pretrained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = np.array(img)
        
        # Check if this is an OpenCV-based augmentation
        if self.aug_name.startswith('opencv_'):
            # Apply OpenCV augmentation
            img = AugmentationLibrary.apply_opencv_augmentation(img, self.aug_name)
            # Resize to target size
            img = cv2.resize(img, (400,400))
            # Convert to tensor manually
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
            # Apply Albumentations augmentation (includes ToTensor)
            augmented = self.aug_transform(image=img)
            img_tensor = augmented['image'].float()

            # Ensure [0, 1] range (ToTensorV2 should handle this, but be safe)
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
        
        # Normalize
        # for i in range(3):
        #     img_tensor[i] = (img_tensor[i] - self.mean[i]) / self.std[i]
        
        # Apply ImageNet normalization
        img_tensor = self.normalize(img_tensor)

        return img_tensor, self.labels[idx]


# ============================================================================
# 3. TRAINING FUNCTIONS (Simplified versions)
# ============================================================================

def train_one_epoch_aug(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # Check for NaN outputs (indicates numerical instability)
        if torch.isnan(outputs).any():
            print("⚠️  WARNING: NaN detected in outputs! Skipping batch...")
            continue
        loss = loss_fn(outputs, labels)
        # Check for NaN loss
        if torch.isnan(loss):
            print("⚠️  WARNING: NaN detected in loss! Skipping batch...")
            continue
        loss.backward()
        
        # ✅ URGENT FIX: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def validate_one_epoch_aug(model, val_loader, loss_fn, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def train_with_augmentation(model, train_loader, val_loader, optimizer, scheduler, 
                            loss_fn, device, num_epochs=50, early_stop=10, use_gradual_unfreezing=False):
    """
    Training loop with early stopping and optional gradual unfreezing
    
    ⚠️ CRITICAL WARNINGS:
    - For small datasets (<500 images): Keep use_gradual_unfreezing=False
    - Use learning rate 1e-5 or lower for datasets <500 images
    - Use gradient clipping (already implemented below)
    - Keep encoder frozen (freeze_encoder=True) for small datasets
    
    Args:
        use_gradual_unfreezing: If True, uses 3-phase training strategy
            - Phase 1: Train classifier only (encoder frozen)
            - Phase 2: Unfreeze last encoder layers  
            - Phase 3: Fine-tune entire model (if enough epochs)
            
    Note: Gradual unfreezing is DISABLED by default (False) because it requires
          large datasets (>500 images) to work well. For small datasets, keep 
          encoder frozen and use low learning rates.
    """
    best_val_acc = -1
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    
    # Define training phases for gradual unfreezing
    phase1_epochs = min(5, num_epochs // 3) if use_gradual_unfreezing else 0
    phase2_epochs = min(5, num_epochs // 3) if use_gradual_unfreezing else 0
    
    # ============================================================================
    # PHASE 1: Train classifier only (encoder frozen)
    # ============================================================================
    if use_gradual_unfreezing and phase1_epochs > 0:
        print("\n" + "="*70)
        print("📌 PHASE 1: Training classifier only (encoder frozen)")
        print("="*70)
        
        # Freeze encoder (using correct attribute name: encoder)
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        # Recreate optimizer for trainable parameters only
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer.param_groups[0]['lr'],  # Keep original lr
            weight_decay=optimizer.param_groups[0]['weight_decay']
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
        
        for epoch in range(phase1_epochs):
            train_loss, train_acc = train_one_epoch_aug(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_acc = validate_one_epoch_aug(model, val_loader, loss_fn, device)
            
            scheduler.step(val_loss)
            
            print(f'  Epoch {epoch+1}/{phase1_epochs} | '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Track best model
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop:
                print(f"  ⚠️ Early stopping triggered at epoch {epoch+1}")
                break
    
    # ============================================================================
    # PHASE 2: Unfreeze last encoder layers
    # ============================================================================
    if use_gradual_unfreezing and phase2_epochs > 0 and epochs_no_improve < early_stop:
        print("\n" + "="*70)
        print("📌 PHASE 2: Fine-tuning last encoder layers")
        print("="*70)
        
        # Unfreeze last encoder blocks
        # For timm EfficientNet, we'll unfreeze the last blocks
        encoder_layers = list(model.encoder.children())
        if len(encoder_layers) > 2:
            # Unfreeze last 2 blocks if structure allows
            for layer in encoder_layers[-2:]:
                if hasattr(layer, 'parameters'):
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            # If structure is simple, unfreeze all encoder
            for param in model.encoder.parameters():
                param.requires_grad = True
        
        # Different learning rates for encoder and classifier
        base_lr = optimizer.param_groups[0]['lr']
        optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': base_lr * 0.1},  # Lower LR for encoder
            {'params': model.classifier.parameters(), 'lr': base_lr}       # Higher LR for classifier
        ], weight_decay=optimizer.param_groups[0]['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
        
        start_epoch = phase1_epochs
        end_epoch = phase1_epochs + phase2_epochs
        
        for epoch in range(start_epoch, end_epoch):
            train_loss, train_acc = train_one_epoch_aug(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_acc = validate_one_epoch_aug(model, val_loader, loss_fn, device)
            
            scheduler.step(val_loss)
            
            print(f'  Epoch {epoch+1}/{end_epoch} | '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop:
                print(f"  ⚠️ Early stopping triggered at epoch {epoch+1}")
                break
    
    # ============================================================================
    # PHASE 3: Fine-tune entire model (remaining epochs)
    # ============================================================================
    remaining_epochs = num_epochs - (phase1_epochs + phase2_epochs)
    
    if remaining_epochs > 0 and epochs_no_improve < early_stop:
        if use_gradual_unfreezing:
            print("\n" + "="*70)
            print("📌 PHASE 3: Fine-tuning entire model")
            print("="*70)
            
            # Unfreeze all encoder layers
            for param in model.encoder.parameters():
                param.requires_grad = True
            
            # Very low learning rate for full model
            base_lr = optimizer.param_groups[0]['lr']
            optimizer = optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': base_lr * 0.01},  # Very low LR
                {'params': model.classifier.parameters(), 'lr': base_lr * 0.1}  # Moderate LR
            ], weight_decay=optimizer.param_groups[0]['weight_decay'])
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
        
        start_epoch = phase1_epochs + phase2_epochs
        
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = train_one_epoch_aug(model, train_loader, optimizer, loss_fn, device)
            val_loss, val_acc = validate_one_epoch_aug(model, val_loader, loss_fn, device)
            
            scheduler.step(val_loss)
            
            phase_name = "PHASE 3" if use_gradual_unfreezing else "Training"
            print(f'  Epoch {epoch+1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop:
                print(f"  ⚠️ Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model with validation accuracy: {best_val_acc:.4f}")
    
    return best_val_acc


def sensitivity_specificity(y_true, y_pred, labels=None):
    """
    Calculate sensitivity (recall) and specificity per class
    
    Sensitivity = TP / (TP + FN) = Ability to detect positive cases
    Specificity = TN / (TN + FP) = Ability to detect negative cases
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    num_classes = cm.shape[0]
    sensitivities = []
    specificities = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return sensitivities, specificities


def evaluate_with_augmentation(model, val_loader, device):
    """Evaluate model and return metrics including sensitivity and specificity"""
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    confidences = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    confidences = np.array(confidences)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    
    # Calculate sensitivity and specificity
    sensitivities, specificities = sensitivity_specificity(y_true, y_pred, labels=[0, 1, 2])
    
    # Confidence analysis
    avg_confidence = float(np.mean(confidences))
    high_conf_mask = confidences > 0.7
    high_conf_acc = accuracy_score(
        y_true[high_conf_mask], 
        y_pred[high_conf_mask]
    ) if high_conf_mask.sum() > 0 else 0.0
    
    # Per-class confidence
    per_class_conf = []
    for cls in [0, 1, 2]:
        cls_mask = (y_true == cls)
        if cls_mask.sum() > 0:
            per_class_conf.append(float(np.mean(confidences[cls_mask])))
        else:
            per_class_conf.append(0.0)
    
    return {
        'accuracy': acc,
        'mcc': mcc,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'sensitivity': sensitivities,
        'specificity': specificities,
        'avg_confidence': avg_confidence,
        'high_confidence_accuracy': high_conf_acc,
        'high_confidence_ratio': float(high_conf_mask.sum() / len(y_true)),
        'per_class_confidence': per_class_conf
    }


# ============================================================================
# 4. MAIN AUGMENTATION EVALUATOR
# ============================================================================

class AugmentationEvaluator:
    """Evaluates each augmentation technique individually"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.output_dir = os.path.join(config.logs_dir_path, 'augmentation_evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate_single_augmentation(self, aug_name):
        """Evaluate a single augmentation technique across all folds"""
        print(f"\n{'='*80}")
        print(f"Evaluating Augmentation: {aug_name.upper()}")
        print(f"{'='*80}\n")
        
        fold_results = []
        
        for fold in range(self.config.no_of_folds):
            print(f"\n--- Fold {fold + 1}/{self.config.no_of_folds} ---")
            
            # Get train/val splits
            train_image_paths, val_image_paths, train_labels, val_labels = get_train_test(fold)
            
            # Get augmentation transform
            aug_transform = AugmentationLibrary.get_augmentation_transform(aug_name, img_size=400)
            
            # Compute normalization
            # Since you use pretrained=True, use ImageNet stats
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            
            # Create datasets
            train_dataset = AugmentedALSDataset(
                train_image_paths, train_labels, aug_transform, aug_name=aug_name
            )
            val_dataset = AugmentedALSDataset(
                val_image_paths, val_labels,
                AugmentationLibrary.get_augmentation_transform('none', 400),
                aug_name='none'
            )
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Compute class weights
            class_weights = compute_class_weights(train_labels, num_classes=3)
            class_weights = class_weights.to(self.config.device)
            
            # Create model
            model = EfficientNetB0Classifier(
                num_classes=3, pretrained=True, freeze_encoder=True, dropout=0.3
            ).to(self.config.device)
            
            # Setup training
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5,min_lr=1e-4, verbose=True)
            
            # Train
            best_val_acc = train_with_augmentation(
                model, train_loader, val_loader, optimizer, scheduler,
                loss_fn, self.config.device, self.config.no_of_epoch, self.config.early_stop, use_gradual_unfreezing=False
            )
            
            # Evaluate
            metrics = evaluate_with_augmentation(model, val_loader, self.config.device)
            
            fold_results.append({
                'fold': fold + 1,
                'val_accuracy': best_val_acc,
                **metrics
            })
            
            print(f"Fold {fold + 1} Results: Accuracy: {best_val_acc:.4f}, MCC: {metrics['mcc']:.4f}")
            print(f"  Sensitivity: {metrics['sensitivity']}")
            print(f"  Specificity: {metrics['specificity']}")

            # Clean up
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
        
        # Aggregate results
        avg_accuracy = np.mean([r['val_accuracy'] for r in fold_results])
        std_accuracy = np.std([r['val_accuracy'] for r in fold_results])
        avg_mcc = np.mean([r['mcc'] for r in fold_results])
        avg_precision = np.mean([r['precision'] for r in fold_results], axis=0)
        avg_recall = np.mean([r['recall'] for r in fold_results], axis=0)
        avg_f1 = np.mean([r['f1'] for r in fold_results], axis=0)
        avg_sensitivity = np.mean([r['sensitivity'] for r in fold_results], axis=0)
        avg_specificity = np.mean([r['specificity'] for r in fold_results], axis=0)
        
        result = {
            'augmentation': aug_name,
            'avg_accuracy': float(avg_accuracy),
            'std_accuracy': float(std_accuracy),
            'avg_mcc': float(avg_mcc),
            'per_class_precision': avg_precision.tolist(),
            'per_class_recall': avg_recall.tolist(),
            'per_class_f1': avg_f1.tolist(),
            'per_class_sensitivity': avg_sensitivity.tolist(),
            'per_class_specificity': avg_specificity.tolist(),
            'fold_results': fold_results
        }
        
        self.results[aug_name] = result
        
        print(f"\n{'='*80}")
        print(f"Overall Results for {aug_name.upper()}:")
        print(f"  Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  MCC: {avg_mcc:.4f}")
        print(f"  Per-Class Sensitivity: Control={avg_sensitivity[0]:.4f}, "
              f"Concordant={avg_sensitivity[1]:.4f}, Discordant={avg_sensitivity[2]:.4f}")
        print(f"  Per-Class Specificity: Control={avg_specificity[0]:.4f}, "
              f"Concordant={avg_specificity[1]:.4f}, Discordant={avg_specificity[2]:.4f}")
        print(f"{'='*80}\n")
        
        # Save intermediate results
        self.save_results()
        
        return result
    
    def evaluate_all_augmentations(self, aug_list=None):
        """Evaluate all or specified augmentations"""
        if aug_list is None:
            aug_list = AugmentationLibrary.get_all_augmentation_names()
        
        for aug_name in aug_list:
            try:
                self.evaluate_single_augmentation(aug_name)
            except Exception as e:
                print(f"❌ Error with augmentation {aug_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    def save_results(self):
        """Save results to JSON"""
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"✅ Results saved to {results_path}")
    
    def print_formatted_results(self):
        """Print results in a clear, formatted table to console and save to text file"""
        if not self.results:
            print("⚠️ No results to display")
            return
        
        class_names = ['Control', 'Concordant', 'Discordant']
        
        # Prepare header
        header = f"{'Augmentation':<20} | {'Accuracy':<18} | "
        for cls in class_names:
            header += f"{cls}_Sens | {cls}_Spec | "
        
        separator = "-" * len(header)
        
        # Build output lines
        output_lines = []
        output_lines.append("\n" + "="*100)
        output_lines.append("AUGMENTATION EVALUATION RESULTS - MEDICAL FOCUS")
        output_lines.append("="*100 + "\n")
        output_lines.append(header)
        output_lines.append(separator)
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['avg_accuracy'],
            reverse=True
        )
        
        # Add data rows
        for aug_name, result in sorted_results:
            # Mark baseline
            aug_display = f"{aug_name} (baseline)" if aug_name == 'none' else aug_name
            
            line = f"{aug_display:<20} | "
            line += f"{result['avg_accuracy']:.4f} ± {result['std_accuracy']:.4f} | "
            
            for cls_idx in range(3):
                sens = result['per_class_sensitivity'][cls_idx]
                spec = result['per_class_specificity'][cls_idx]
                line += f"{sens:.4f}    | {spec:.4f}    | "
            
            output_lines.append(line)
        
        output_lines.append(separator)
        
        # Add summary statistics
        output_lines.append("\n" + "="*100)
        output_lines.append("SUMMARY: AVERAGE SENSITIVITY & SPECIFICITY PER AUGMENTATION")
        output_lines.append("="*100 + "\n")
        
        summary_header = f"{'Augmentation':<20} | {'Avg_Sensitivity':<15} | {'Avg_Specificity':<15} | {'Accuracy':<10}"
        output_lines.append(summary_header)
        output_lines.append("-" * len(summary_header))
        
        for aug_name, result in sorted_results:
            avg_sens = sum(result['per_class_sensitivity']) / 3
            avg_spec = sum(result['per_class_specificity']) / 3
            
            aug_display = f"{aug_name} (baseline)" if aug_name == 'none' else aug_name
            
            summary_line = f"{aug_display:<20} | {avg_sens:<15.4f} | {avg_spec:<15.4f} | {result['avg_accuracy']:<10.4f}"
            output_lines.append(summary_line)
        
        output_lines.append("\n" + "="*100)
        output_lines.append("INTERPRETATION GUIDE")
        output_lines.append("="*100)
        output_lines.append("Sensitivity: Of all actual cases, how many did we detect? (Higher = Better)")
        output_lines.append("Specificity: Of all non-cases, how many did we correctly identify? (Higher = Better)")
        output_lines.append("For medical diagnosis: HIGH sensitivity is critical to avoid missing cases!")
        output_lines.append("="*100 + "\n")
        
        # Print to console
        for line in output_lines:
            print(line)
        
        # Save to text file
        txt_path = os.path.join(self.output_dir, 'results_summary.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:  # ✅ URGENT FIX: Added utf-8 encoding
            f.write('\n'.join(output_lines))
        
        print(f"✅ Formatted results saved to: {txt_path}\n")
        
        # Also create a simplified markdown table
        self._create_markdown_table()
    
    def _create_markdown_table(self):
        """Create a markdown-formatted table for easy copy-paste into papers"""
        class_names = ['Control', 'Concordant', 'Discordant']
        
        md_lines = []
        md_lines.append("\n# Augmentation Evaluation Results\n")
        md_lines.append("## Main Results Table\n")
        md_lines.append("\n**Legend:**\n")
        md_lines.append("- 🟦 Albumentations (Tier 1: Must Have)\n")
        md_lines.append("- 🟪 Albumentations (Tier 2: Highly Recommended)\n")
        md_lines.append("- 🟥 Albumentations (Original)\n")
        md_lines.append("- 🟣 **OpenCV-Based Standard Techniques** (for comparison)\n")
        md_lines.append("- ⚫ Baseline\n\n")
        
        # Header
        header = "| Augmentation | Type | Accuracy | "
        for cls in class_names:
            header += f"{cls} Sens | {cls} Spec | "
        header += "Avg Sens | Avg Spec |"
        
        separator = "|" + "|".join(["---" for _ in range(3 + len(class_names)*2 + 2)]) + "|"
        
        md_lines.append(header)
        md_lines.append(separator)
        
        # Helper function to get augmentation type
        def get_aug_type(aug_name):
            if aug_name == 'none':
                return 'Baseline'
            elif aug_name.startswith('opencv_'):
                return '**OpenCV**'
            elif aug_name in ['horizontal_flip', 'vertical_flip', 'random_rotation', 
                             'elastic_deformation', 'brightness_contrast', 'random_zoom']:
                return 'Alb-T1'
            elif aug_name in ['clahe', 'gaussian_noise', 'gaussian_blur', 'color_jitter']:
                return 'Alb-T2'
            elif aug_name == 'grayscale':
                return 'Alb-Orig'
            else:
                return 'Alb-Other'
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['avg_accuracy'],
            reverse=True
        )
        
        # Data rows
        for aug_name, result in sorted_results:
            avg_sens = sum(result['per_class_sensitivity']) / 3
            avg_spec = sum(result['per_class_specificity']) / 3
            
            aug_type = get_aug_type(aug_name)
            row = f"| {aug_name} | {aug_type} | {result['avg_accuracy']:.4f} ± {result['std_accuracy']:.4f} | "
            
            for cls_idx in range(3):
                sens = result['per_class_sensitivity'][cls_idx]
                spec = result['per_class_specificity'][cls_idx]
                row += f"{sens:.4f} | {spec:.4f} | "
            
            row += f"{avg_sens:.4f} | {avg_spec:.4f} |"
            md_lines.append(row)
        
        # Save markdown
        md_path = os.path.join(self.output_dir, 'results_table.md')
        with open(md_path, 'w', encoding='utf-8') as f:  # ✅ URGENT FIX: Added utf-8 encoding
            f.write('\n'.join(md_lines))
        
        print(f"✅ Markdown table saved to: {md_path}")
    
    def create_comparison_table(self):
        """Create CSV comparison table with sensitivity and specificity"""
        data = []
        class_names = ['Control', 'Concordant', 'Discordant']
        
        for aug_name, result in self.results.items():
            row = {
                'Augmentation': aug_name,
                'Accuracy': f"{result['avg_accuracy']:.4f} ± {result['std_accuracy']:.4f}",
                'MCC': f"{result['avg_mcc']:.4f}",
            }
            
            # Add per-class metrics
            for i, class_name in enumerate(class_names):
                row[f'{class_name}_Sensitivity'] = f"{result['per_class_sensitivity'][i]:.4f}"
                row[f'{class_name}_Specificity'] = f"{result['per_class_specificity'][i]:.4f}"
                row[f'{class_name}_F1'] = f"{result['per_class_f1'][i]:.4f}"
            
            data.append(row)
        
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.output_dir, 'augmentation_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Comparison table saved to {csv_path}")
        
        # Also create a separate detailed metrics table
        self._create_detailed_metrics_table()
        
        return df
    
    def _create_detailed_metrics_table(self):
        """Create detailed per-class metrics table"""
        class_names = ['Control', 'Concordant', 'Discordant']
        
        for cls_idx, class_name in enumerate(class_names):
            data = []
            for aug_name, result in self.results.items():
                row = {
                    'Augmentation': aug_name,
                    'Sensitivity': f"{result['per_class_sensitivity'][cls_idx]:.4f}",
                    'Specificity': f"{result['per_class_specificity'][cls_idx]:.4f}",
                    'Precision': f"{result['per_class_precision'][cls_idx]:.4f}",
                    'Recall': f"{result['per_class_recall'][cls_idx]:.4f}",
                    'F1-Score': f"{result['per_class_f1'][cls_idx]:.4f}",
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            csv_path = os.path.join(self.output_dir, f'{class_name}_detailed_metrics.csv')
            df.to_csv(csv_path, index=False)
            print(f"✅ {class_name} detailed metrics saved to {csv_path}")
    
    def plot_results(self):
        """Create visualization plots including sensitivity/specificity"""
        if not self.results:
            print("⚠️ No results to plot")
            return
        
        # Plot 1: Accuracy comparison
        self._plot_accuracy_comparison()
        
        # Plot 2: Sensitivity bar chart (NEW!)
        self._plot_sensitivity_bar_chart()
        
        # Plot 3: Specificity bar chart (NEW!)
        self._plot_specificity_bar_chart()
        
        # Plot 4: Combined Sens + Spec bar chart (NEW!)
        self._plot_combined_sens_spec_bars()
        
        # Plot 5: Sensitivity and Specificity heatmaps
        self._plot_sensitivity_specificity_heatmaps()
        
        # Plot 6: Per-class performance comparison
        self._plot_per_class_comparison()
    
    def _plot_sensitivity_bar_chart(self):
        """Bar chart comparing average sensitivity across augmentations"""
        aug_names = list(self.results.keys())
        avg_sensitivities = [
            sum(self.results[aug]['per_class_sensitivity']) / 3 
            for aug in aug_names
        ]
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(aug_names)), avg_sensitivities)
        
        # Color code
        colors = []
        for aug in aug_names:
            if aug == 'none':
                colors.append('gray')
            else:
                colors.append('steelblue')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xticks(range(len(aug_names)), aug_names, rotation=45, ha='right')
        plt.ylabel('Average Sensitivity', fontsize=12)
        plt.title('Average Sensitivity Comparison Across Augmentations', 
                  fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        
        if 'none' in self.results:
            baseline_sens = sum(self.results['none']['per_class_sensitivity']) / 3
            plt.axhline(y=baseline_sens, color='red', linestyle='--', 
                       linewidth=2, label=f'Baseline: {baseline_sens:.3f}')
            plt.legend()
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'sensitivity_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Sensitivity comparison plot saved to {plot_path}")
    
    def _plot_specificity_bar_chart(self):
        """Bar chart comparing average specificity across augmentations"""
        aug_names = list(self.results.keys())
        avg_specificities = [
            sum(self.results[aug]['per_class_specificity']) / 3 
            for aug in aug_names
        ]
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(aug_names)), avg_specificities)
        
        # Color code
        colors = []
        for aug in aug_names:
            if aug == 'none':
                colors.append('gray')
            else:
                colors.append('coral')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xticks(range(len(aug_names)), aug_names, rotation=45, ha='right')
        plt.ylabel('Average Specificity', fontsize=12)
        plt.title('Average Specificity Comparison Across Augmentations', 
                  fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        
        if 'none' in self.results:
            baseline_spec = sum(self.results['none']['per_class_specificity']) / 3
            plt.axhline(y=baseline_spec, color='red', linestyle='--', 
                       linewidth=2, label=f'Baseline: {baseline_spec:.3f}')
            plt.legend()
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'specificity_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Specificity comparison plot saved to {plot_path}")
    
    def _plot_combined_sens_spec_bars(self):
        """Side-by-side bars comparing sensitivity and specificity"""
        aug_names = list(self.results.keys())
        avg_sensitivities = [
            sum(self.results[aug]['per_class_sensitivity']) / 3 
            for aug in aug_names
        ]
        avg_specificities = [
            sum(self.results[aug]['per_class_specificity']) / 3 
            for aug in aug_names
        ]
        
        x = np.arange(len(aug_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(16, 7))
        
        bars1 = ax.bar(x - width/2, avg_sensitivities, width, 
                       label='Sensitivity', color='steelblue')
        bars2 = ax.bar(x + width/2, avg_specificities, width, 
                       label='Specificity', color='coral')
        
        ax.set_xlabel('Augmentation', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Sensitivity vs Specificity Comparison', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(aug_names, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add baseline lines
        if 'none' in self.results:
            baseline_sens = sum(self.results['none']['per_class_sensitivity']) / 3
            baseline_spec = sum(self.results['none']['per_class_specificity']) / 3
            ax.axhline(y=baseline_sens, color='steelblue', linestyle='--', 
                      linewidth=1, alpha=0.5)
            ax.axhline(y=baseline_spec, color='coral', linestyle='--', 
                      linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'sensitivity_specificity_bars.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Combined sens/spec bar chart saved to {plot_path}")
    
    def _plot_accuracy_comparison(self):
        """Bar plot of accuracy for each augmentation"""
        aug_names = list(self.results.keys())
        accuracies = [self.results[aug]['avg_accuracy'] for aug in aug_names]
        stds = [self.results[aug]['std_accuracy'] for aug in aug_names]
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(aug_names)), accuracies, yerr=stds, capsize=5)
        
        # Color code
        colors = []
        for aug in aug_names:
            if aug == 'none':
                colors.append('gray')
            elif aug in ['horizontal_flip', 'vertical_flip', 'random_rotation', 
                         'elastic_deformation', 'brightness_contrast', 'random_zoom']:
                colors.append('green')
            elif aug in ['clahe', 'gaussian_noise', 'gaussian_blur', 'color_jitter']:
                colors.append('blue')
            elif aug == 'grayscale':
                colors.append('red')
            elif aug.startswith('opencv_'):
                colors.append('magenta')  # Unique color for OpenCV-based standard augmentations
            else:
                colors.append('orange')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xticks(range(len(aug_names)), aug_names, rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.title('Augmentation Evaluation - Accuracy Comparison')
        
        # Create legend for color coding
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', label='Baseline'),
            Patch(facecolor='green', label='Albumentations - Tier 1 (Must Have)'),
            Patch(facecolor='blue', label='Albumentations - Tier 2 (Highly Recommended)'),
            Patch(facecolor='red', label='Albumentations - Original Grayscale'),
            Patch(facecolor='magenta', label='OpenCV - Standard Techniques'),
            Patch(facecolor='orange', label='Albumentations - Other')
        ]
        
        if 'none' in self.results:
            plt.axhline(y=self.results['none']['avg_accuracy'], color='black', 
                        linestyle='--', label='Baseline Accuracy', linewidth=2)
        
        plt.legend(handles=legend_elements, loc='lower right', fontsize=8)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'accuracy_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Accuracy comparison plot saved to {plot_path}")
    
    def _plot_sensitivity_specificity_heatmaps(self):
        """Heatmaps of sensitivity and specificity per class"""
        aug_names = list(self.results.keys())
        class_names = ['Control', 'Concordant', 'Discordant']
        
        # Prepare data
        sens_matrix = []
        spec_matrix = []
        
        for aug in aug_names:
            sens_matrix.append(self.results[aug]['per_class_sensitivity'])
            spec_matrix.append(self.results[aug]['per_class_specificity'])
        
        sens_matrix = np.array(sens_matrix)
        spec_matrix = np.array(spec_matrix)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, len(aug_names) * 0.4))
        
        # Sensitivity heatmap
        sns.heatmap(
            sens_matrix, 
            annot=True, 
            fmt='.3f',
            xticklabels=class_names,
            yticklabels=aug_names,
            cmap='RdYlGn',
            vmin=0, vmax=1,
            cbar_kws={'label': 'Sensitivity'},
            ax=axes[0]
        )
        axes[0].set_title('Sensitivity per Class and Augmentation', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Augmentation', fontsize=12)
        
        # Specificity heatmap
        sns.heatmap(
            spec_matrix, 
            annot=True, 
            fmt='.3f',
            xticklabels=class_names,
            yticklabels=aug_names,
            cmap='RdYlGn',
            vmin=0, vmax=1,
            cbar_kws={'label': 'Specificity'},
            ax=axes[1]
        )
        axes[1].set_title('Specificity per Class and Augmentation', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('Augmentation', fontsize=12)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'sensitivity_specificity_heatmaps.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Sensitivity/Specificity heatmaps saved to {plot_path}")
    
    def _plot_per_class_comparison(self):
        """Bar plots comparing sensitivity and specificity across augmentations for each class"""
        aug_names = list(self.results.keys())
        class_names = ['Control', 'Concordant', 'Discordant']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for cls_idx, class_name in enumerate(class_names):
            sensitivities = [self.results[aug]['per_class_sensitivity'][cls_idx] for aug in aug_names]
            specificities = [self.results[aug]['per_class_specificity'][cls_idx] for aug in aug_names]
            
            x = np.arange(len(aug_names))
            width = 0.35
            
            axes[cls_idx].bar(x - width/2, sensitivities, width, label='Sensitivity', color='steelblue')
            axes[cls_idx].bar(x + width/2, specificities, width, label='Specificity', color='coral')
            
            axes[cls_idx].set_xlabel('Augmentation', fontsize=10)
            axes[cls_idx].set_ylabel('Score', fontsize=10)
            axes[cls_idx].set_title(f'{class_name} Class', fontsize=12, fontweight='bold')
            axes[cls_idx].set_xticks(x)
            axes[cls_idx].set_xticklabels(aug_names, rotation=45, ha='right', fontsize=8)
            axes[cls_idx].set_ylim([0, 1])
            axes[cls_idx].legend()
            axes[cls_idx].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Per-Class Sensitivity and Specificity Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'per_class_sens_spec_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Per-class comparison plot saved to {plot_path}")


# ============================================================================
# 5. MAIN EXECUTION FUNCTION
# ============================================================================

def run_augmentation_evaluation(quick_test=False):
    """
    Main function to run augmentation evaluation
    
    Args:
        quick_test: If True, only tests 2 augmentations (baseline + horizontal_flip)
    """
    
    if not ALBUMENTATIONS_AVAILABLE:
        print("\n❌ ERROR: albumentations not installed!")
        print("Install with: pip install albumentations")
        return
    
    print("\n" + "="*80)
    print("AUGMENTATION EVALUATION SYSTEM")
    print("="*80 + "\n")
    
    evaluator = AugmentationEvaluator(config)
    
    if quick_test:
        print("🔹 Mode: QUICK TEST (2 augmentations)")
        print("   Testing: baseline (none) + horizontal_flip")
        print("   Estimated time: 1-2 hours with GPU\n")
        aug_list = ['none', 'horizontal_flip']
    else:
        print("🔹 Mode: FULL EVALUATION (all augmentations)")
        print(f"   Testing: {len(AugmentationLibrary.get_all_augmentation_names())} augmentations")
        print("   Estimated time: 10-20 hours with GPU\n")
        aug_list = None
    
    # Run evaluation
    evaluator.evaluate_all_augmentations(aug_list)
    
    # Generate outputs
    evaluator.create_comparison_table()
    evaluator.print_formatted_results()  # ← NEW: Print formatted results
    evaluator.plot_results()
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print(f"Results saved to: {evaluator.output_dir}")
    print(f"Check: {os.path.join(evaluator.output_dir, 'results_summary.txt')}")  # ← Point to summary
    print("="*80 + "\n")


# ============================================================================
# END OF FILE
# ============================================================================