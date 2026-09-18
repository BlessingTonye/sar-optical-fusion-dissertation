import torch
import numpy as np
from sklearn.metrics import f1_score


def bootstrap(model, test_ds, device):
    """
    Collects per-patch predictions and reference labels from a trained
    single-input model while keeping patches separate for subsequent
    bootstrap comparison.

    Args:
        model (nn.Module): Trained model to evaluate.
        test_ds (Dataset): Test dataset containing image and label patches.
        device (torch.device): Device used for model inference.

    Returns:
        tuple: Per-patch predictions and corresponding reference labels.
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
    Collects per-patch predictions and reference labels from a trained
    dual-input model while keeping patches separate for subsequent
    bootstrap comparison.

    Args:
        model (nn.Module): Trained dual-input model to evaluate.
        test_ds (Dataset): Test dataset containing corresponding SAR,
            optical, and label patches.
        device (torch.device): Device used for model inference.

    Returns:
        tuple: Per-patch predictions and corresponding reference labels.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img_sar, img_optical, lbl in test_ds:
            img_sar = img_sar.to(device)
            img_optical = img_optical.to(device)
            preds = model(img_sar.unsqueeze(0), img_optical.unsqueeze(0)).argmax(dim=1).cpu().numpy().flatten()
            labels = lbl.numpy().flatten()
            all_preds.append(preds)
            all_labels.append(labels)

    return all_preds, all_labels


def paired_bootstrap(preds_a, preds_b, labels, name_a, name_b, n_iterations=5000,
                     n_classes=4, lower=2.5, upper=97.5, random_state=42):
    """
    Performs a paired patch-level bootstrap comparison of two models
    using macro-F1. The same patch indices are resampled for both
    models in every bootstrap replicate.

    Args:
        preds_a (list): Per-patch predictions from model A.
        preds_b (list): Per-patch predictions from model B.
        labels (list): Per-patch reference labels shared by both models.
        name_a (str): Name of model A used in printed output.
        name_b (str): Name of model B used in printed output.
        n_iterations (int): Number of bootstrap replicates.
        n_classes (int): Number of segmentation classes.
        lower (float): Lower percentile used for the confidence interval.
        upper (float): Upper percentile used for the confidence interval.
        random_state (int): Random seed used for reproducibility.

    Returns:
        tuple: Mean macro-F1 difference, confidence interval, and p-value.
    """
    if not (len(preds_a) == len(preds_b) == len(labels)):
        raise ValueError("Both models and labels must contain the same number of patches.")

    rng = np.random.default_rng(random_state)
    n_patches = len(labels)
    differences = []

    for _ in range(n_iterations):
        indices = rng.choice(n_patches, size=n_patches, replace=True)

        flat_preds_a = np.concatenate([preds_a[i] for i in indices])
        flat_preds_b = np.concatenate([preds_b[i] for i in indices])
        flat_labels = np.concatenate([labels[i] for i in indices])

        f1_a = f1_score(flat_labels, flat_preds_a, labels=list(range(n_classes)), average='macro')
        f1_b = f1_score(flat_labels, flat_preds_b, labels=list(range(n_classes)), average='macro')

        differences.append(f1_a - f1_b)

    differences = np.array(differences)

    mean_difference = np.mean(differences)
    ci_lower, ci_upper = np.percentile(differences, [lower, upper])
    p_value = 2 * min((differences < 0).mean(), (differences > 0).mean())

    significant = "SIGNIFICANT" if (ci_lower > 0 or ci_upper < 0) else "not significant"

    print(f"{name_a} vs {name_b}")
    print(f"  Mean difference: {mean_difference:+.4f}")
    print(f"  95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  p-value: {p_value:.4f}  ({significant})")

    return mean_difference, (ci_lower, ci_upper), p_value