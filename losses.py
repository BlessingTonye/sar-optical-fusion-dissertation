import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Computes soft Dice loss for multi-class segmentation, measuring
    overlap between predicted and true class regions per class.

    Args:
        n_classes (int): number of segmentation classes.
        smooth (float): small constant to prevent division by zero.
    """
    def __init__(self, n_classes, smooth=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): raw model output, shape (batch, n_classes, H, W).
            targets (Tensor): true class indices, shape (batch, H, W), with -1 for invalid pixels.

        Returns:
            Tensor: scalar loss value.
        """
        valid_mask = (targets != -1)
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(safe_targets, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        mask = valid_mask.unsqueeze(1).float()
        probs = probs * mask
        targets_onehot = targets_onehot * mask
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_per_class = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """
    Combines weighted cross-entropy and soft Dice loss for training
    under class imbalance.

    Args:
        class_weights (Tensor): inverse-frequency weights per class, applied to the cross-entropy term.
        n_classes (int): number of segmentation classes.
        ce_weight (float): weighting of the cross-entropy term in the combined loss.
        dice_weight (float): weighting of the Dice term in the combined loss.
    """
    def __init__(self, class_weights, n_classes, ce_weight=0.6, dice_weight=0.4):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.dice_loss = SoftDiceLoss(n_classes=n_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): raw model output, shape (batch, n_classes, H, W).
            targets (Tensor): true class indices, shape (batch, H, W), with -1 for invalid pixels.

        Returns:
            Tensor: scalar combined loss value.
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice