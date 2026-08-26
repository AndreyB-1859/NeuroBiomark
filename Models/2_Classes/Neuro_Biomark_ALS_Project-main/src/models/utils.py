import os.path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from contourpy import contour_generator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
from config.config import config
from src.process_data.load_data import get_dataloaders_and_classweights
from src.models.DenseNet121 import AttentionDenseNet121
from torch.nn.functional import one_hot
from scipy.stats import f_oneway, t, sem

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        # one_hot_list = [one_hot(l, num_classes=len(loss_fn.weight)) for l in labels]
        # labels = torch.stack(one_hot_list, dim=0)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc

def validate_one_epoch(model, val_loader, loss_fn, device):
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

def train_loop(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, num_epoch, early_stop):

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    best_val_acc = -1
    best_val_loss = float("inf")
    epochs_no_improve = 0   # counter for early stopping

    print("\ttraining the model.")
    for epoch in range(num_epoch):
        print(f"\tEpoch: {epoch+1}/{num_epoch}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        # safe validation call
        try:
            val_loss, val_acc = validate_one_epoch(model, val_loader, loss_fn, device)
        except Exception as e:
            print(f"\tValidation failed at epoch {epoch+1}: {e}")
            break        
            
        scheduler.step(val_loss)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)


        # string the best performing model.
        model_path = os.path.join(config.saved_models_dir_path,f"fold_{config.fold_no}_model_weights.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),model_path)
        
        # ---- Early stopping logic (based on validation loss) ----
        if val_loss < best_val_loss - 1e-4:  # improvement threshold
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"\t\tNo improvement for {epochs_no_improve} epoch(s)")

        # Stop if patience exceeded
        if epochs_no_improve >= early_stop:
            print(f"\tEarly stopping at epoch {epoch+1}")
            break

        print(f"\t\tTrainLoss: {train_loss:.4f}, TrainAccuracy: {train_acc:.4f} ValLoss: {val_loss:.4f}, ValAccuracy: {val_acc:.4f}")

    # storing the results in file
    results_path = os.path.join(config.logs_dir_path, f"fold_{config.fold_no}_training_results.csv")
    print(f"[DEBUG] train_loss_list={len(train_loss_list)}, "
      f"train_acc_list={len(train_acc_list)}, "
      f"val_loss_list={len(val_loss_list)}, "
      f"val_acc_list={len(val_acc_list)}")
    
    # --- after loop ---
    # truncate lists to the same length (final safety)
    min_len = min(len(train_loss_list), len(train_acc_list), len(val_loss_list), len(val_acc_list))

    epochs_ran = len(train_loss_list)
    
    training_results = pd.DataFrame({
        "epoch": list(range(1,min_len+1)),
        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "val_loss": val_loss_list,
        "val_acc": val_acc_list
    })
    training_results.to_csv(results_path, index=False)
    print(f"For fold_{config.fold_no}: Early stopped after {epochs_ran} epochs out of {num_epoch}")


def sensitivity_specificity(y_true, y_pred, labels=None):
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

def evaluate_model(fold, val_loader):
    print("\tEvaluating the model")

    model_path = os.path.join(config.saved_models_dir_path, f"fold_{config.fold_no}_model_weights.pth")
    model = AttentionDenseNet121().to(config.device)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    sens, spec = sensitivity_specificity(y_true, y_pred, labels=[0,1,2])
    
    #saving evaluation results
    results_path = os.path.join(config.logs_dir_path, f"fold_{config.fold_no}_evaluation_results.csv")

    # Create a DataFrame for saving
    results_df = pd.DataFrame({
        'Class': [0,1,2],
        'Sensitivity': sens,
        'Specificity': spec
    })

    # Add overall metrics
    overall_metrics = pd.DataFrame({
        'Class': ['Overall'],
        'Sensitivity': [np.nan],
        'Specificity': [np.nan],
        'Accuracy': [acc],
        'MCC': [mcc]
    })

    results_df = pd.concat([results_df, overall_metrics], ignore_index=True)
    results_df.to_csv(results_path, index=False)
    

def run_folds(fold):

    train_loader, val_loader, class_weights =  get_dataloaders_and_classweights(fold=fold)
    class_weights = class_weights.to(config.device)

    model = AttentionDenseNet121(freeze_until='denseblock4').to(config.device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    threshold=1e-4,
    cooldown=1,
    min_lr=1e-6)

    train_loop(model, train_loader, val_loader, optimizer, scheduler, loss_fn, config.device, config.no_of_epoch, config.early_stop)
    evaluate_model(fold, val_loader)
    
def mean_confidence_interval(data, confidence=0.95):
    """Compute mean and 95% confidence interval."""
    data = np.array(data)
    data = data[~np.isnan(data)]
    n = len(data)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    m = np.mean(data)
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2, n - 1)
    return m, (m - h, m + h)


def anova_test_multi_class(num_folds=15, save_plot=True):
    class_labels = [0, 1]  # 0 = Control, 1 = ALS-Concordant, 2 = ALS-Discordant
    class_names = ["ALS-Concordant", "ALS-Discordant"]

    sensitivities = {cls: [] for cls in class_labels}
    specificities = {cls: [] for cls in class_labels}

    # Load fold-wise results
    for fold in range(num_folds):
        eval_path = os.path.join(config.logs_dir_path, f"fold_{fold}_evaluation_results.csv")
        if not os.path.exists(eval_path):
            print(f"Missing file for fold {fold}: {eval_path}")
            continue

        df = pd.read_csv(eval_path)
        for cls in class_labels:
            row = df[df['Class'].astype(str) == str(cls)]
            if not row.empty:
                sensitivities[cls].append(row['Sensitivity'].values[0])
                specificities[cls].append(row['Specificity'].values[0])

    # Perform ANOVA
    sens_groups = [sensitivities[c] for c in class_labels if len(sensitivities[c]) > 0]
    spec_groups = [specificities[c] for c in class_labels if len(specificities[c]) > 0]

    f_sens, p_sens = f_oneway(*sens_groups)
    f_spec, p_spec = f_oneway(*spec_groups)

    print("\n=== ANOVA Results Across Classes ===")
    print(f"Sensitivity: F = {f_sens:.4f}, p = {p_sens:.4f}")
    print(f"Specificity: F = {f_spec:.4f}, p = {p_spec:.4f}")

    # Confidence intervals and means
    stats_summary = []
    for cls in class_labels:
        mean_s, ci_s = mean_confidence_interval(sensitivities[cls])
        mean_p, ci_p = mean_confidence_interval(specificities[cls])
        stats_summary.append({
            "Class": class_names[cls],
            "Sens_Mean": mean_s, "Sens_CI_Low": ci_s[0], "Sens_CI_High": ci_s[1],
            "Spec_Mean": mean_p, "Spec_CI_Low": ci_p[0], "Spec_CI_High": ci_p[1]
        })

    stats_df = pd.DataFrame(stats_summary)
    print("\n=== Class-wise Means & 95% Confidence Intervals ===")
    print(stats_df)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    x = np.arange(len(class_names))

    # Sensitivity plot
    sens_means = stats_df["Sens_Mean"]
    sens_err = [
        stats_df["Sens_Mean"] - stats_df["Sens_CI_Low"],
        stats_df["Sens_CI_High"] - stats_df["Sens_Mean"]
    ]
    axes[0].bar(x, sens_means, yerr=sens_err, capsize=6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=15)
    axes[0].set_title("Sensitivity (Mean ± 95% CI)")
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_ylim(0, 1)

    # Specificity plot
    spec_means = stats_df["Spec_Mean"]
    spec_err = [
        stats_df["Spec_Mean"] - stats_df["Spec_CI_Low"],
        stats_df["Spec_CI_High"] - stats_df["Spec_Mean"]
    ]
    axes[1].bar(x, spec_means, yerr=spec_err, capsize=6, color="orange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=15)
    axes[1].set_title("Specificity (Mean ± 95% CI)")
    axes[1].set_ylabel("Specificity")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # Optionally save figure
    if save_plot:
        plot_path = os.path.join(config.logs_dir_path, "anova_visualization.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {plot_path}")

    return {
        "anova": {"sensitivity": (f_sens, p_sens), "specificity": (f_spec, p_spec)},
        "summary": stats_df
    }
