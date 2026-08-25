import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix, jaccard_score


def evaluate_macro_f1(model, loader, n_classes, device):
    """
    Computes validation macro-F1 for a model, used for checkpoint
    selection and early stopping during training.

    Args:
        model (nn.Module): trained or in-training model to evaluate.
        loader (DataLoader): validation data loader.
        n_classes (int): number of segmentation classes.
        device (torch.device): device to run evaluation on.

    Returns:
        float: macro-averaged F1 score across all classes.
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
    Computes a full suite of evaluation metrics (overall accuracy,
    Cohen's Kappa, macro-F1, macro-precision, macro-recall, mean IoU)
    plus per-class breakdowns and a confusion matrix.

    Args:
        model (nn.Module): trained model to evaluate.
        loader (DataLoader): test/validation data loader.
        n_classes (int): number of segmentation classes.
        class_names (dict): mapping of class index to class name.
        device (torch.device): device to run evaluation on.

    Returns:
        tuple: (results dict, per-class F1, per-class precision,
                per-class recall, confusion matrix array).
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
        'overall_accuracy': accuracy_score(y_true, y_pred),  # % of all pixels correctly classified
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),  # accuracy adjusted for chance agreement
        'macro_f1': f1_score(y_true, y_pred, labels=labels, average='macro'),  # F1 averaged equally across classes
        'macro_precision': precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),  # correctness of positive predictions, averaged across classes
        'macro_recall': recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),  # coverage of actual positives found, averaged across classes
        'mean_iou': jaccard_score(y_true, y_pred, labels=labels, average='macro'),  # overlap between predicted and true regions, averaged across classes
    }

    # same four metrics, computed separately per class instead of averaged
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)  # rows = true class, columns = predicted class

    return results, per_class_f1, per_class_precision, per_class_recall, cm


def evaluate_full_metrics_dual(model, loader, n_classes, class_names, device):
    """
    Computes a full suite of evaluation metrics (overall accuracy,
    Cohen's Kappa, macro-F1, macro-precision, macro-recall, mean IoU)
    plus per-class breakdowns and a confusion matrix, for a two-input
    (dual-encoder) model such as MiddleFusionUNet.

    Args:
        model (nn.Module): trained two-input model to evaluate.
        loader (DataLoader): test/validation data loader yielding
            (img_s1, img_s2, label) per batch.
        n_classes (int): number of segmentation classes.
        class_names (dict): mapping of class index to class name.
        device (torch.device): device to run evaluation on.

    Returns:
        tuple: (results dict, per-class F1, per-class precision,
                per-class recall, confusion matrix array).
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for img_s1, img_s2, lbls in loader:
            img_s1 = img_s1.to(device)
            img_s2 = img_s2.to(device)
            preds = model(img_s1, img_s2).argmax(dim=1).cpu().numpy().flatten()
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


def plot_confusion_matrices_for_dissertation(cm1, cm2, class_names, title1, title2, normalize=True):
    """
    Plots two confusion matrices side by side, for direct comparison
    between two models.

    Args:
        cm1 (ndarray): confusion matrix for the first model.
        cm2 (ndarray): confusion matrix for the second model.
        class_names (dict): mapping of class index to class name.
        title1 (str): subplot title for the first matrix.
        title2 (str): subplot title for the second matrix.
        normalize (bool): whether to display row-wise percentages
            instead of raw counts.
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
    Computes a pixel-wise correct/incorrect map by comparing a
    model's prediction against ground truth.

    Args:
        true_label (ndarray): ground truth label array.
        pred (ndarray): predicted label array, same shape as true_label.

    Returns:
        ndarray: binary array, 1 where correct, 0 where incorrect.
    """
    return (pred == true_label).astype(int)


def plot_error_map_comparison(true_label, pred1, pred2, class_names, title1, title2):
    """
    Plots ground truth, prediction, and a correct/incorrect error map
    for two models on the same test patch, for direct comparison.

    Args:
        true_label (ndarray): ground truth label array for the patch.
        pred1 (ndarray): predicted label array for the first model.
        pred2 (ndarray): predicted label array for the second model.
        class_names (dict): mapping of class index to class name.
        title1 (str): row label for the first model.
        title2 (str): row label for the second model.
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
    plt.savefig('/content/error_map_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()