import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def compute_patch_proportions(labels, num_classes=4):
    """
    Calculates the proportion of pixels belonging to each class for
    every extracted label patch.

    Args:
        labels (list): List of label patches with shape (H, W).
        num_classes (int): Number of segmentation classes.

    Returns:
        ndarray: Per-patch class proportions with shape
            (n_patches, num_classes).
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
    Groups patches into spatial blocks and calculates the mean class
    proportions for each block.

    Args:
        positions (list): List of patch positions as (row, column) coordinates.
        proportions (ndarray): Per-patch class proportions from
            compute_patch_proportions().
        block_size (int): Spatial block size in pixels.

    Returns:
        tuple: Patch DataFrame with block IDs and block-level class
            proportions.
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
    Splits spatial blocks into training, validation, and test sets
    using multi-label stratified shuffle splitting based on class
    presence within each block.

    Args:
        block_props (DataFrame): Mean class proportions for each spatial block.
        test_size (float): Fraction of blocks assigned to the test split.
        val_size (float): Fraction of blocks assigned to the validation split.
        class_threshold (float): Minimum class proportion required for a class
            to be considered present in a block.
        random_state (int): Random seed used for reproducibility.

    Returns:
        tuple: Training, validation, and test block ID arrays.
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
    Assigns each patch to the training, validation, or test split
    according to its spatial block.

    Args:
        patches_df (DataFrame): Patch-level DataFrame containing a block_id column.
        train_blocks (array-like): Block IDs assigned to the training split.
        val_blocks (array-like): Block IDs assigned to the validation split.
        test_blocks (array-like): Block IDs assigned to the test split.

    Returns:
        DataFrame: Patch DataFrame with an additional split column.
    """
    split_map = {block: 'train' for block in train_blocks}
    split_map.update({block: 'val' for block in val_blocks})
    split_map.update({block: 'test' for block in test_blocks})

    patches_df['split'] = patches_df['block_id'].map(split_map)
    return patches_df


def compute_class_weights(patches_df, num_classes=4):
    """
    Calculates inverse frequency class weights from the class
    proportions in the training split for use in the weighted loss
    function.

    Args:
        patches_df (DataFrame): Patch-level DataFrame containing the class
            proportion columns for the training split.
        num_classes (int): Number of segmentation classes.

    Returns:
        ndarray: Inverse frequency weight for each segmentation class.
    """
    class_cols = [f'class_{c}_prop' for c in range(num_classes)]
    mean_class_freq = patches_df[class_cols].mean().to_numpy()
    weights = 1.0 / (mean_class_freq + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights