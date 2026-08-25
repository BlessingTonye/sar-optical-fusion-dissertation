import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def compute_patch_proportions(labels, num_classes=4):
    """
    Computes the proportion of pixels belonging to each class,
    for every patch.

    Args:
        labels (list): list of label patches (arrays), each shape (H, W).
        num_classes (int): number of segmentation classes.

    Returns:
        ndarray: shape (n_patches, num_classes), per-patch class proportions.
    """
    proportion_matrix = []
    for lbl in labels:
        valid_pixels = lbl[~np.isnan(lbl)]
        total_valid = len(valid_pixels)
        if total_valid == 0:
            props = [0.0] * num_classes
        else:
            counts = [np.sum(valid_pixels == c) for c in range(num_classes)]
            props = [count / total_valid for count in counts]
        proportion_matrix.append(props)
    return np.array(proportion_matrix)


def compute_block_proportions(positions, proportions, block_size=512):
    """
    Groups patches into spatial blocks and computes the mean
    class proportion within each block.

    Args:
        positions (list): list of (row, col) patch positions.
        proportions (ndarray): per-patch class proportions from compute_patch_proportions().
        block_size (int): block size in pixels.

    Returns:
        tuple: (patches_df with block_id assigned per patch,
                block_props DataFrame with mean class proportions per block).
    """
    rows = np.array([p[0] for p in positions])
    cols = np.array([p[1] for p in positions])
    block_ids = (rows // block_size).astype(str) + '_' + (cols // block_size).astype(str)

    patches_df = pd.DataFrame({
        'patch_id': range(len(positions)),
        'row': rows,
        'col': cols,
        'block_id': block_ids,
        'class_0_prop': proportions[:, 0],
        'class_1_prop': proportions[:, 1],
        'class_2_prop': proportions[:, 2],
        'class_3_prop': proportions[:, 3],
    })

    class_columns = ['class_0_prop', 'class_1_prop', 'class_2_prop', 'class_3_prop']
    block_props = patches_df.groupby('block_id')[class_columns].mean()
    return patches_df, block_props


def stratified_block_split(block_props, test_size=0.15, val_size=0.15, class_threshold=0.005, random_state=42):
    """
    Splits spatial blocks into train/val/test using multi-label
    stratified shuffle splitting on class-presence indicators.

    Args:
        block_props (DataFrame): mean class proportions per block.
        test_size (float): fraction of blocks assigned to test.
        val_size (float): fraction of blocks assigned to validation.
        class_threshold (float): minimum proportion for a class to
            count as "present" in a block.
        random_state (int): random seed for reproducibility.

    Returns:
        tuple: (train_blocks, val_blocks, test_blocks) block ID arrays.
    """
    y_blocks = (block_props.values >= class_threshold).astype(int)
    x_blocks = block_props.index.values

    splitter_1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter_1.split(x_blocks, y_blocks))

    splitter_2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    train_idx, val_idx = next(splitter_2.split(x_blocks[train_val_idx], y_blocks[train_val_idx]))

    train_blocks = x_blocks[train_val_idx][train_idx]
    val_blocks = x_blocks[train_val_idx][val_idx]
    test_blocks = x_blocks[test_idx]

    return train_blocks, val_blocks, test_blocks


def assign_patch_splits(patches_df, train_blocks, val_blocks, test_blocks):
    """
    Maps each patch to its split (train/val/test) based on which
    spatial block it belongs to.

    Args:
        patches_df (DataFrame): patch-level DataFrame with a block_id column.
        train_blocks (array-like): block IDs assigned to training.
        val_blocks (array-like): block IDs assigned to validation.
        test_blocks (array-like): block IDs assigned to test.

    Returns:
        DataFrame: patches_df with a new 'split' column added.
    """
    split_map = {block: 'train' for block in train_blocks}
    split_map.update({block: 'val' for block in val_blocks})
    split_map.update({block: 'test' for block in test_blocks})

    patches_df['split'] = patches_df['block_id'].map(split_map)
    return patches_df


def compute_class_weights(patches_df, num_classes=4):
    """
    Computes inverse-frequency class weights from per-patch class
    proportions, for use in a weighted loss function under class
    imbalance.

    Args:
        patches_df (DataFrame): patch-level DataFrame with columns
            'class_0_prop' ... 'class_{num_classes-1}_prop', typically
            filtered to the training split only.
        num_classes (int): number of segmentation classes.

    Returns:
        ndarray: shape (num_classes,), inverse-frequency weight per class.
    """
    class_cols = [f'class_{c}_prop' for c in range(num_classes)]
    mean_class_freq = patches_df[class_cols].mean().to_numpy()
    weights = 1.0 / (mean_class_freq + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights