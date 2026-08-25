import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])


class PatchDataset(Dataset):
    """
    Wraps extracted patches into a PyTorch dataset, with optional
    channel slicing (for single-sensor configurations) and augmentation.

    Args:
        image_patches (list): full list of extracted image patches.
        label_patches (list): full list of corresponding label patches.
        patch_ids (array-like): indices into image_patches/label_patches for this split.
        channel_slice (slice, optional): which channels to select, e.g. slice(0,2) for S1-only.
        transform (callable, optional): augmentation to apply; None for validation/test.
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
        Args:
            idx (int): index within this split.

        Returns:
            tuple: (image tensor, label tensor).
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
    Pairs two single-sensor PatchDataset instances so they can be
    consumed together by a dual-input model's training loop.

    Args:
        ds_s1 (Dataset): PatchDataset built with channel_slice for Sentinel-1.
        ds_s2 (Dataset): PatchDataset built with channel_slice for Sentinel-2.
    """
    def __init__(self, ds_s1, ds_s2):
        self.ds_s1 = ds_s1
        self.ds_s2 = ds_s2

    def __len__(self):
        return len(self.ds_s1)

    def __getitem__(self, idx):
        img_s1, lbl = self.ds_s1[idx]
        img_s2, _ = self.ds_s2[idx]
        return img_s1, img_s2, lbl