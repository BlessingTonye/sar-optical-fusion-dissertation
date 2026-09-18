import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix, jaccard_score


def evaluate_macro_f1(model, loader, n_classes, device):
    """
    Calculates macro-F1 on the validation dataset for checkpoint
    selection and early stopping during model training.

    Args:
        model (nn.Module): Segmentation model to evaluate.
        loader (DataLoader): DataLoader containing the validation dataset.
        n_classes (int): Number of segmentation classes.
        device (torch.device): Device used for model evaluation.

    Returns:
        float: Macro-F1 score across all segmentation classes.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy().flatten()
            targets = lbls.numpy().flatten()
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    valid_mask = all_targets != -1
    return f1_score(all_targets[valid_mask], all_preds[valid_mask], labels=list(range(n_classes)), average='macro')


def evaluate_full_metrics(model, loader, n_classes, class_names, device):
    """
    Calculates the overall and per-class evaluation metrics for a
    single-input segmentation model, together with the confusion matrix.

    Args:
        model (nn.Module): Trained segmentation model to evaluate.
        loader (DataLoader): DataLoader containing the validation or test dataset.
        n_classes (int): Number of segmentation classes.
        class_names (dict): Mapping of class indices to class names.
        device (torch.device): Device used for model evaluation.

    Returns:
        tuple: Overall metrics, per-class F1, per-class precision, per-class recall, and confusion matrix.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy().flatten()
            targets = lbls.numpy().flatten()
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    valid_mask = all_targets != -1
    y_true, y_pred = all_targets[valid_mask], all_preds[valid_mask]

    labels = list(range(n_classes))

    results = {
        'overall_accuracy': accuracy_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred), 
        'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro'),
        'macro_precision': precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),  
        'macro_recall': recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),  
        'mean_iou': jaccard_score(y_true, y_pred, labels=labels, average='macro'),  
    }

    # Calculate class-specific metrics without macro-averaging
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)  

    return results, per_class_f1, per_class_precision, per_class_recall, cm


def evaluate_full_metrics_dual(model, loader, n_classes, class_names, device):
    """
    Calculates the overall and per-class evaluation metrics for the
    dual-input Middle Fusion model, together with the confusion matrix.

    Args:
        model (nn.Module): Trained dual-input segmentation model to evaluate.
        loader (DataLoader): DataLoader containing corresponding SAR and optical inputs with their labels.
        n_classes (int): Number of segmentation classes.
        class_names (dict): Mapping of class indices to class names.
        device (torch.device): Device used for model evaluation.

    Returns:
        tuple: Overall metrics, per-class F1, per-class precision, per-class recall, and confusion matrix.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for img_sar, img_optical, lbls in loader:
            img_sar = img_sar.to(device)
            img_optical = img_optical.to(device)
            preds = model(img_sar, img_optical).argmax(dim=1).cpu().numpy().flatten()
            targets = lbls.numpy().flatten()
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    valid_mask = all_targets != -1
    y_true, y_pred = all_targets[valid_mask], all_preds[valid_mask]

    labels = list(range(n_classes))

    results = {
        'overall_accuracy': accuracy_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro'),
        'macro_precision': precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'mean_iou': jaccard_score(y_true, y_pred, labels=labels, average='macro'),
    }

    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return results, per_class_f1, per_class_precision, per_class_recall, cm


def plot_confusion_matrices(cm1, cm2, class_names, title1, title2, normalize=True):
    """
    Plots two confusion matrices side by side to support direct
    comparison between two segmentation models.

    Args:
        cm1 (ndarray): Confusion matrix for the first model.
        cm2 (ndarray): Confusion matrix for the second model.
        class_names (dict): Mapping of class indices to class names.
        title1 (str): Title for the first confusion matrix.
        title2 (str): Title for the second confusion matrix.
        normalize (bool): Whether to display row-wise proportions instead of raw counts.
    """
    if normalize:
        cm1 = cm1.astype('float') / cm1.sum(axis=1, keepdims=True)
        cm2 = cm2.astype('float') / cm2.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    labels = list(class_names.values())

    for ax, cm, title in zip(axes, [cm1, cm2], [title1, title2]):
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar_kws={'label': 'Proportion of true class' if normalize else 'Count', 'shrink': 0.85},
                    annot_kws={"size": 11}, square=True)
        ax.set_xlabel('Predicted class', fontsize=11)
        ax.set_ylabel('True class', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', labelsize=9.5)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.subplots_adjust(wspace=1.2)
    plt.savefig('/content/confusion_matrices_dissertation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def compute_error_map(true_label, pred):
    """
    Creates a pixel-level error map by comparing the predicted and
    reference class labels.

    Args:
        true_label (ndarray): Reference label array.
        pred (ndarray): Predicted label array with the same shape as the reference labels.

    Returns:
        ndarray: Binary array where 1 indicates a correct prediction and 0 indicates an incorrect prediction.
    """
    return (pred == true_label).astype(int)


def plot_error_map_comparison(true_label, pred1, pred2, class_names, title1, title2):
    """
    Displays the reference labels, model predictions, and pixel-level
    error maps for two models on the same test patch.

    Args:
        true_label (ndarray): Reference label array for the test patch.
        pred1 (ndarray): Predicted label array from the first model.
        pred2 (ndarray): Predicted label array from the second model.
        class_names (dict): Mapping of class indices to class names.
        title1 (str): Name displayed for the first model.
        title2 (str): Name displayed for the second model.
    """
    class_colors = ['darkgreen', 'yellowgreen', 'blue', 'gray']
    cmap_classes = mcolors.ListedColormap(class_colors)
    cmap_error = mcolors.ListedColormap(['red', 'lightgreen'])

    error_map1 = compute_error_map(true_label, pred1)
    error_map2 = compute_error_map(true_label, pred2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(true_label, cmap=cmap_classes, vmin=0, vmax=3)
    axes[0, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[0, 1].imshow(pred1, cmap=cmap_classes, vmin=0, vmax=3)
    axes[0, 1].set_title(f"{title1} Prediction", fontsize=12, fontweight='bold')
    axes[0, 2].imshow(error_map1, cmap=cmap_error, vmin=0, vmax=1)
    axes[0, 2].set_title(f"{title1}: Correct vs Incorrect", fontsize=12, fontweight='bold')

    axes[1, 0].imshow(true_label, cmap=cmap_classes, vmin=0, vmax=3)
    axes[1, 0].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[1, 1].imshow(pred2, cmap=cmap_classes, vmin=0, vmax=3)
    axes[1, 1].set_title(f"{title2} Prediction", fontsize=12, fontweight='bold')
    axes[1, 2].imshow(error_map2, cmap=cmap_error, vmin=0, vmax=1)
    axes[1, 2].set_title(f"{title2}: Correct vs Incorrect", fontsize=12, fontweight='bold')

    for row in axes:
        for ax in row:
            ax.axis('off')

    class_patches = [mpatches.Patch(color=class_colors[i], label=list(class_names.values())[i]) for i in range(4)]
    error_patches = [mpatches.Patch(color='lightgreen', label='Correct'), mpatches.Patch(color='red', label='Incorrect')]
    fig.legend(handles=class_patches, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=10)
    fig.legend(handles=error_patches, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10)

    plt.tight_layout()
    plt.show()