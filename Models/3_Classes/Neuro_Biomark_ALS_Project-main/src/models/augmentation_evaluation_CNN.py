"""
Complete Augmentation Evaluation System for BasicCNN

This is the main script that evaluates all augmentation techniques using:
- BasicCNN model (optimized for small datasets)
- 5-fold Group CV (prevents patient leakage)
- Per-class metrics (accuracy, sensitivity, specificity)
- Comprehensive visualization
- Best fold logging

Usage:
    from augmentation_evaluation_basiccnn import run_augmentation_evaluation
    run_augmentation_evaluation(quick_test=False)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import your modules
from config.config import config
from src.models.BasicCNN import create_basic_cnn, get_recommended_hyperparameters
from src.process_data.dataLoaderCNN import create_dataloaders_for_fold


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_per_class_metrics(y_true, y_pred, num_classes=3):
    """
    Calculate per-class sensitivity, specificity, precision
    
    Returns dict with:
    - overall_accuracy
    - class_X_sensitivity (recall/TPR)
    - class_X_specificity (TNR)  
    - class_X_precision
    - macro averages
    """
    metrics = {}
    
    # Overall accuracy
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    for class_idx in range(num_classes):
        # Binary: this class vs rest
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_true_binary, y_pred_binary, labels=[0, 1]
        ).ravel()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        metrics[f'class_{class_idx}_sensitivity'] = sensitivity
        metrics[f'class_{class_idx}_specificity'] = specificity
        metrics[f'class_{class_idx}_precision'] = precision
    
    # Macro averages
    metrics['macro_sensitivity'] = np.mean([
        metrics[f'class_{i}_sensitivity'] for i in range(num_classes)
    ])
    metrics['macro_specificity'] = np.mean([
        metrics[f'class_{i}_specificity'] for i in range(num_classes)
    ])
    
    return metrics


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.long())
        
        # Check for NaN
        if torch.isnan(loss):
            print("⚠️ NaN loss detected, skipping batch")
            continue
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


# ============================================================================
# SINGLE FOLD TRAINING
# ============================================================================

def train_single_fold(fold_num, aug_name, device, hyperparams, save_dir):
    """
    Train model for one fold with specified augmentation
    
    Returns:
        dict with best_metrics and training history
    """
    
    print(f"\n{'='*70}")
    print(f"Training Fold {fold_num+1}/5 - Augmentation: {aug_name}")
    print(f"{'='*70}")
    
    # Create dataloaders (now returns normalization stats too)
    train_loader, val_loader, class_weights, (mean, std) = create_dataloaders_for_fold(
        fold_num=fold_num,
        aug_name=aug_name,
        batch_size=hyperparams['batch_size'],
        num_workers=2
    )
    
    # Create model with 400x400 input size
    model = create_basic_cnn(num_classes=3, dropout=0.5, image_size=400)
    model = model.to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=hyperparams['scheduler_params']['factor'],
        patience=hyperparams['scheduler_params']['patience'],
        min_lr=hyperparams['scheduler_params']['min_lr'],
        verbose=False
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_metrics = {}
    
    # Training loop
    for epoch in range(hyperparams['max_epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Calculate detailed metrics
        detailed_metrics = calculate_per_class_metrics(val_labels, val_preds)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{hyperparams['max_epochs']} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best metrics
            best_metrics = {
                'fold': fold_num,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                **detailed_metrics
            }
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                save_dir, f'fold{fold_num}_best.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'metrics': best_metrics
            }, checkpoint_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= hyperparams['early_stopping_patience']:
            print(f"\n⏸️  Early stopping at epoch {epoch+1}")
            print(f"Best epoch: {best_epoch+1} | Best val_loss: {best_val_loss:.4f}")
            break
    
    print(f"\n✅ Fold {fold_num+1} complete!")
    print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    return {
        'best_metrics': best_metrics,
        'history': history,
        'best_epoch': best_epoch
    }


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def run_augmentation_evaluation(quick_test=False):
    """
    Main function to evaluate all augmentation techniques
    
    Args:
        quick_test: If True, test only 3 augmentations (~30 min)
                   If False, test all augmentations (~10-15 hours)
    """
    
    print("\n" + "="*80)
    print("AUGMENTATION EVALUATION - BASICCNN")
    print("="*80)
    print(f"Dataset: 190 images (70 Control, 60 ALS+D, 60 ALS-D)")
    print(f"Model: BasicCNN (~150K parameters)")
    print(f"Cross-validation: 5-fold Group CV")
    print(f"Mode: {'Quick Test (3 augmentations)' if quick_test else 'Full Evaluation (22 augmentations)'}")
    print("="*80 + "\n")
    
    # Setup
    device = config.device
    hyperparams = get_recommended_hyperparameters()
    
    # Define augmentations
    if quick_test:
        augmentations = ['none', 'horizontal_flip', 'brightness_contrast']
        print("⚡ QUICK TEST MODE")
    else:
        augmentations = [
            'none',  # Baseline
            'horizontal_flip', 'vertical_flip', 'random_rotation',
            'elastic_deformation', 'brightness_contrast', 'random_zoom',
            'clahe', 'gaussian_noise', 'gaussian_blur', 'color_jitter',
            'shift_scale_rotate', 'grayscale',
            'opencv_flip', 'opencv_rotate_90', 'opencv_grayscale',
            'opencv_gray_denoise', 'opencv_color_denoise',
            'opencv_scale', 'opencv_crop', 'opencv_noise', 'opencv_perspective'
        ]
    
    print(f"\nTesting {len(augmentations)} augmentations × 5 folds = {len(augmentations) * 5} training runs")
    est_time_per_fold = 15 if device.type == 'cuda' else 45
    print(f"Estimated time: {len(augmentations) * 5 * est_time_per_fold / 60:.1f} hours")
    print("="*80 + "\n")
    
    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        config.logs_dir_path, 
        'augmentation_evaluation',
        f'basiccnn_{timestamp}'
    )
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📁 Results directory: {results_dir}\n")
    
    # Store results
    all_results = []
    best_fold_logs = {}
    
    # Evaluate each augmentation
    total_start = time.time()
    
    for aug_idx, aug_name in enumerate(augmentations):
        print(f"\n{'#'*80}")
        print(f"AUGMENTATION {aug_idx+1}/{len(augmentations)}: {aug_name.upper()}")
        print(f"{'#'*80}")
        
        aug_start = time.time()
        
        # Create directory for this augmentation
        aug_dir = os.path.join(results_dir, aug_name)
        os.makedirs(aug_dir, exist_ok=True)
        
        # Run 5-fold CV
        fold_results = []
        fold_histories = []
        
        for fold in range(config.no_of_folds):
            result = train_single_fold(
                fold_num=fold,
                aug_name=aug_name,
                device=device,
                hyperparams=hyperparams,
                save_dir=aug_dir
            )
            
            fold_results.append(result['best_metrics'])
            fold_histories.append(result['history'])
        
        # Aggregate metrics
        aggregate = {
            'augmentation': aug_name,
            'mean_val_acc': np.mean([f['val_acc'] for f in fold_results]),
            'std_val_acc': np.std([f['val_acc'] for f in fold_results]),
            'mean_val_loss': np.mean([f['val_loss'] for f in fold_results]),
            'std_val_loss': np.std([f['val_loss'] for f in fold_results]),
        }
        
        # Per-class metrics
        for class_idx in range(3):
            aggregate[f'mean_class_{class_idx}_sensitivity'] = np.mean([
                f[f'class_{class_idx}_sensitivity'] for f in fold_results
            ])
            aggregate[f'std_class_{class_idx}_sensitivity'] = np.std([
                f[f'class_{class_idx}_sensitivity'] for f in fold_results
            ])
            aggregate[f'mean_class_{class_idx}_specificity'] = np.mean([
                f[f'class_{class_idx}_specificity'] for f in fold_results
            ])
            aggregate[f'std_class_{class_idx}_specificity'] = np.std([
                f[f'class_{class_idx}_specificity'] for f in fold_results
            ])
        
        # Macro averages
        aggregate['mean_macro_sensitivity'] = np.mean([
            f['macro_sensitivity'] for f in fold_results
        ])
        aggregate['mean_macro_specificity'] = np.mean([
            f['macro_specificity'] for f in fold_results
        ])
        
        all_results.append(aggregate)
        
        # Best fold
        best_fold_idx = np.argmax([f['val_acc'] for f in fold_results])
        best_fold_logs[aug_name] = {
            'fold': int(best_fold_idx),
            'history': fold_histories[best_fold_idx],
            'metrics': fold_results[best_fold_idx]
        }
        
        aug_time = time.time() - aug_start
        print(f"\n✅ {aug_name} complete in {aug_time/60:.1f} minutes")
        print(f"   Mean Val Acc: {aggregate['mean_val_acc']:.4f} ± {aggregate['std_val_acc']:.4f}")
        print(f"   Best Fold: {best_fold_idx+1} with {fold_results[best_fold_idx]['val_acc']:.4f}")
    
    total_time = time.time() - total_start
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")
    
    # Save aggregate results
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(results_dir, 'augmentation_comparison.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"✅ Saved: {results_csv}")
    
    # Save best fold logs
    best_fold_json = os.path.join(results_dir, 'best_fold_logs.json')
    with open(best_fold_json, 'w') as f:
        json.dump(best_fold_logs, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"✅ Saved: {best_fold_json}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_time_hours': total_time / 3600,
        'num_augmentations': len(augmentations),
        'num_folds': config.no_of_folds,
        'device': str(device),
        'hyperparameters': hyperparams,
        'best_augmentation': results_df.loc[results_df['mean_val_acc'].idxmax(), 'augmentation'],
        'best_accuracy': float(results_df['mean_val_acc'].max())
    }
    summary_json = os.path.join(results_dir, 'evaluation_summary.json')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✅ Saved: {summary_json}")
    
    # Generate plots
    print(f"\n📊 Generating visualizations...")
    plot_results(results_df, best_fold_logs, results_dir)
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results: {results_dir}")
    print(f"Best augmentation: {summary['best_augmentation']} ({summary['best_accuracy']:.4f})")
    print(f"{'='*80}\n")
    
    return results_df, best_fold_logs


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results_df, best_fold_logs, save_dir):
    """Generate all visualization plots"""
    
    sns.set_style("whitegrid")
    
    # 1. Overall accuracy
    plot_accuracy_comparison(results_df, save_dir)
    
    # 2. Per-class sensitivity
    plot_per_class_sensitivity(results_df, save_dir)
    
    # 3. Per-class specificity
    plot_per_class_specificity(results_df, save_dir)
    
    # 4. Sensitivity vs Specificity
    plot_sens_spec_scatter(results_df, save_dir)
    
    # 5. Top 5 training curves
    plot_top_training_curves(results_df, best_fold_logs, save_dir)
    
    print("✅ All plots saved!")


def plot_accuracy_comparison(results_df, save_dir):
    """Bar plot of accuracies"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sorted_df = results_df.sort_values('mean_val_acc', ascending=False)
    
    bars = ax.barh(sorted_df['augmentation'], sorted_df['mean_val_acc'],
                   xerr=sorted_df['std_val_acc'], capsize=4)
    
    # Color by accuracy
    colors = plt.cm.RdYlGn(sorted_df['mean_val_acc'].values)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Mean Validation Accuracy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Augmentation', fontsize=13, fontweight='bold')
    ax.set_title('Augmentation Comparison - Overall Accuracy', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        ax.text(row['mean_val_acc'] + 0.01, i, f"{row['mean_val_acc']:.3f}",
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_accuracy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_sensitivity(results_df, save_dir):
    """Per-class sensitivity comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    class_names = ['Control', 'ALS+Dementia', 'ALS-Dementia']
    
    for class_idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        col = f'mean_class_{class_idx}_sensitivity'
        std_col = f'std_class_{class_idx}_sensitivity'
        
        sorted_df = results_df.sort_values(col, ascending=False)
        
        bars = ax.barh(sorted_df['augmentation'], sorted_df[col],
                      xerr=sorted_df[std_col], capsize=3)
        
        colors = plt.cm.Blues(sorted_df[col].values)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Sensitivity', fontsize=12, fontweight='bold')
        if class_idx == 0:
            ax.set_ylabel('Augmentation', fontsize=12, fontweight='bold')
        ax.set_title(class_name, fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Per-Class Sensitivity Comparison', 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_sensitivity_per_class.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_specificity(results_df, save_dir):
    """Per-class specificity comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    class_names = ['Control', 'ALS+Dementia', 'ALS-Dementia']
    
    for class_idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        col = f'mean_class_{class_idx}_specificity'
        std_col = f'std_class_{class_idx}_specificity'
        
        sorted_df = results_df.sort_values(col, ascending=False)
        
        bars = ax.barh(sorted_df['augmentation'], sorted_df[col],
                      xerr=sorted_df[std_col], capsize=3)
        
        colors = plt.cm.Oranges(sorted_df[col].values)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Specificity', fontsize=12, fontweight='bold')
        if class_idx == 0:
            ax.set_ylabel('Augmentation', fontsize=12, fontweight='bold')
        ax.set_title(class_name, fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Per-Class Specificity Comparison',
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_specificity_per_class.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_sens_spec_scatter(results_df, save_dir):
    """Sensitivity vs specificity scatter"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(
        results_df['mean_macro_specificity'],
        results_df['mean_macro_sensitivity'],
        s=250, alpha=0.6,
        c=results_df['mean_val_acc'],
        cmap='viridis',
        edgecolors='black',
        linewidth=2
    )
    
    # Labels
    for _, row in results_df.iterrows():
        ax.annotate(
            row['augmentation'],
            (row['mean_macro_specificity'], row['mean_macro_sensitivity']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.4, linewidth=2, label='Sens = Spec')
    
    ax.set_xlabel('Macro Specificity', fontsize=13, fontweight='bold')
    ax.set_ylabel('Macro Sensitivity', fontsize=13, fontweight='bold')
    ax.set_title('Sensitivity vs Specificity Trade-off',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Validation Accuracy', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_sens_vs_spec.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_training_curves(results_df, best_fold_logs, save_dir, top_n=5):
    """Training curves for top N augmentations"""
    top_augs = results_df.nlargest(top_n, 'mean_val_acc')['augmentation'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, aug_name in enumerate(top_augs):
        if aug_name not in best_fold_logs:
            continue
        
        ax = axes[idx]
        history = best_fold_logs[aug_name]['history']
        fold = best_fold_logs[aug_name]['fold']
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Losses
        ax.plot(epochs, history['train_loss'], 'b-', 
                label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-',
                label='Val Loss', linewidth=2)
        
        # Accuracies on secondary axis
        ax2 = ax.twinx()
        ax2.plot(epochs, history['train_acc'], 'b--',
                label='Train Acc', linewidth=2, alpha=0.6)
        ax2.plot(epochs, history['val_acc'], 'r--',
                label='Val Acc', linewidth=2, alpha=0.6)
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f'{aug_name} (Fold {fold+1})',
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, 
                 loc='upper right', fontsize=9)
    
    if top_n < 6:
        fig.delaxes(axes[-1])
    
    plt.suptitle(f'Training Curves - Top {top_n} Augmentations',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '5_training_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\nStarting augmentation evaluation...")
    print("Use quick_test=True for fast testing (3 augmentations)")
    print("Use quick_test=False for full evaluation (22 augmentations)\n")
    
    # Run evaluation
    results_df, best_fold_logs = run_augmentation_evaluation(quick_test=True)