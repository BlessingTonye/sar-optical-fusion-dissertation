import torch
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import f1_score


def bootstrap(model, test_ds, device):
    """
    Collects per-patch predictions and true labels from a trained
    model, kept separate (not flattened together) for later
    bootstrap resampling.

    Args:
        model (nn.Module): trained model to evaluate.
        test_ds (Dataset): test dataset to draw patches from.
        device (torch.device): device to run inference on.

    Returns:
        tuple: (list of per-patch predictions, list of per-patch true labels).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img, lbl in test_ds:
            img = img.to(device)
            preds = model(img.unsqueeze(0)).argmax(dim=1).cpu().numpy().flatten()
            labels = lbl.numpy().flatten()
            all_preds.append(preds)
            all_labels.append(labels)

    return all_preds, all_labels


def bootstrap_dual(model, test_ds, device):
    """
    Collects per-patch predictions and true labels from a trained
    two-input (dual-encoder) model, such as MiddleFusionUNet, kept
    separate (not flattened together) for later bootstrap resampling.

    Args:
        model (nn.Module): trained two-input model to evaluate.
        test_ds (Dataset): test dataset yielding (img_s1, img_s2, label) per item.
        device (torch.device): device to run inference on.

    Returns:
        tuple: (list of per-patch predictions, list of per-patch true labels).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img_s1, img_s2, lbl in test_ds:
            img_s1 = img_s1.to(device)
            img_s2 = img_s2.to(device)
            preds = model(img_s1.unsqueeze(0), img_s2.unsqueeze(0)).argmax(dim=1).cpu().numpy().flatten()
            labels = lbl.numpy().flatten()
            all_preds.append(preds)
            all_labels.append(labels)

    return all_preds, all_labels


def resampling(pred, label, n_iterations=5000, n_classes=4):
    """
    Performs patch-level paired bootstrap resampling to build a
    distribution of macro-F1 scores under repeated resampling.

    Args:
        pred (list): per-patch predictions from bootstrap().
        label (list): per-patch true labels from bootstrap().
        n_iterations (int): number of bootstrap replicates.
        n_classes (int): number of segmentation classes.

    Returns:
        list: macro-F1 score for each replicate.
    """
    bootstrapped_f1s = []

    for _ in range(n_iterations):
        resampled_pred, resampled_label = resample(pred, label, replace=True)

        flat_pred = np.concatenate(resampled_pred)
        flat_label = np.concatenate(resampled_label)

        score = f1_score(flat_label, flat_pred, labels=list(range(n_classes)), average='macro')
        bootstrapped_f1s.append(score)

    return bootstrapped_f1s


def compare_models(f1_scores_a, f1_scores_b, name_a, name_b, lower, upper):
    """
    Compares two models' bootstrap replicate score distributions,
    computing the mean difference, a confidence interval, and a
    two-sided p-value.

    Args:
        f1_scores_a (list): replicate scores for model A.
        f1_scores_b (list): replicate scores for model B.
        name_a (str): label for model A, used in printed output.
        name_b (str): label for model B, used in printed output.
        lower (float): lower percentile for the confidence interval (e.g. 2.5).
        upper (float): upper percentile for the confidence interval (e.g. 97.5).

    Returns:
        tuple: (mean difference, (CI lower, CI upper), p-value).
    """
    diffs = np.array(f1_scores_a) - np.array(f1_scores_b)

    observed_diff = np.mean(diffs)
    ci_lower, ci_upper = np.percentile(diffs, [lower, upper])
    p_value = 2 * min((diffs < 0).mean(), (diffs > 0).mean())

    significant = "SIGNIFICANT" if (ci_lower > 0 or ci_upper < 0) else "not significant"

    print(f"{name_a} vs {name_b}")
    print(f"  Mean difference: {observed_diff:+.4f}")
    print(f"  CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  p-value: {p_value:.4f}  ({significant})")

    return observed_diff, (ci_lower, ci_upper), p_value