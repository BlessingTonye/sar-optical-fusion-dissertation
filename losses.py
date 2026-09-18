import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Calculates soft Dice loss for multi-class semantic segmentation
    by measuring the overlap between predicted and reference class
    regions.

    Args:
        n_classes (int): Number of segmentation classes.
        smooth (float): Small constant used to prevent division by zero.
    """
    def __init__(self, n_classes, smooth=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Calculates soft Dice loss while excluding invalid pixels.

        Args:
            logits (Tensor): Raw model outputs with shape
                (batch, n_classes, H, W).
            targets (Tensor): Reference class indices with shape
                (batch, H, W), where -1 represents invalid pixels.

        Returns:
            Tensor: Scalar soft Dice loss.
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
    Combines weighted cross-entropy loss and soft Dice loss to account
    for class imbalance while also considering class-region overlap.

    Args:
        class_weights (Tensor): Class weights applied to the cross-entropy loss.
        n_classes (int): Number of segmentation classes.
        ce_weight (float): Weight assigned to the cross-entropy loss.
        dice_weight (float): Weight assigned to the soft Dice loss.
    """
    def __init__(self, class_weights, n_classes, ce_weight=0.6, dice_weight=0.4):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.dice_loss = SoftDiceLoss(n_classes=n_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        """
        Calculates the weighted combination of cross-entropy and
        soft Dice loss.

        Args:
            logits (Tensor): Raw model outputs with shape
                (batch, n_classes, H, W).
            targets (Tensor): Reference class indices with shape
                (batch, H, W), where -1 represents invalid pixels.

        Returns:
            Tensor: Scalar combined loss.
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice