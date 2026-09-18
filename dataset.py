import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])


class PatchDataset(Dataset):
    """
    Creates a PyTorch dataset from extracted image and label patches.
    It supports channel selection for single-sensor configurations
    and optional data augmentation.

    Args:
        image_patches (list): List of extracted image patches.
        label_patches (list): List of corresponding label patches.
        patch_ids (array-like): Indices of the patches included in the
            dataset split.
        channel_slice (slice, optional): Channels to select from each
            image patch, such as slice(0, 2) for SAR-only input.
        transform (callable, optional): Data augmentation applied to
            image and label patches. None for validation and testing.
    """
    def __init__(self, image_patches, label_patches, patch_ids, channel_slice=None, transform=None):
        self.image_patches = image_patches
        self.label_patches = label_patches
        self.patch_ids = patch_ids
        self.channel_slice = channel_slice
        self.transform = transform

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        """
        Retrieves an image patch and its corresponding label.

        Args:
            idx (int): Index of the patch within the dataset split.

        Returns:
            tuple: Image tensor and corresponding label tensor.
        """
        pid = self.patch_ids[idx]

        img = np.asarray(self.image_patches[pid])
        if self.channel_slice is not None:
            img = img[self.channel_slice]

        img = np.nan_to_num(img, nan=0.0)

        lbl = np.squeeze(np.asarray(self.label_patches[pid]))
        lbl = np.where(np.isnan(lbl), -1, lbl).astype(np.int64)

        if self.transform is not None:
            img_hwc = np.transpose(img, (1, 2, 0))
            augmented = self.transform(image=img_hwc, mask=lbl)
            img_hwc = augmented['image']
            lbl = augmented['mask']
            img = np.transpose(img_hwc, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)


class PairedDataset(Dataset):
    """
    Creates a paired dataset containing corresponding SAR and optical
    inputs for the dual-input Middle Fusion model. Data augmentation is
    applied jointly before separating the SAR and optical channels to
    preserve spatial alignment.

    Args:
        image_patches (list): List of fused SAR and optical image patches.
        label_patches (list): List of corresponding label patches.
        patch_ids (array-like): Indices of the patches included in the
            dataset split.
        n_sar_bands (int): Number of SAR input channels.
        transform (callable, optional): Data augmentation applied jointly
            to the image and label patches.
    """
    def __init__(self, image_patches, label_patches, patch_ids, n_sar_bands, transform=None):
        self.image_patches = image_patches
        self.label_patches = label_patches
        self.patch_ids = patch_ids
        self.n_sar_bands = n_sar_bands
        self.transform = transform

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        """
        Retrieves corresponding SAR, optical, and label patches.

        Args:
            idx (int): Index of the patch within the dataset split.

        Returns:
            tuple: SAR tensor, optical tensor, and corresponding label tensor.
        """
        pid = self.patch_ids[idx]

        img = np.asarray(self.image_patches[pid])
        img = np.nan_to_num(img, nan=0.0)

        lbl = np.squeeze(np.asarray(self.label_patches[pid]))
        lbl = np.where(np.isnan(lbl), -1, lbl).astype(np.int64)

        if self.transform is not None:
            img_hwc = np.transpose(img, (1, 2, 0))
            augmented = self.transform(image=img_hwc, mask=lbl)
            img_hwc = augmented['image']
            lbl = augmented['mask']
            img = np.transpose(img_hwc, (2, 0, 1))

        img_sar = img[:self.n_sar_bands]
        img_optical = img[self.n_sar_bands:]

        return (
            torch.tensor(img_sar, dtype=torch.float32),
            torch.tensor(img_optical, dtype=torch.float32),
            torch.tensor(lbl, dtype=torch.long)
        )