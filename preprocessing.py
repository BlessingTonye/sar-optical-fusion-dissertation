import numpy as np
import rasterio


def file_upload(file_path):
    """
    Loads a GeoTIFF file into a NumPy array and converts the NoData
    value to NaN for subsequent processing.

    Args:
        file_path (str): Path to the GeoTIFF file.

    Returns:
        tuple: Image array and corresponding Rasterio profile.
    """
    with rasterio.open(file_path) as src:
        image = src.read().astype('float32')
        profile = src.profile.copy()
        profile.update(dtype='float32', nodata=np.nan)

        image[image == -9999] = np.nan

    return image, profile


def percentile_normalise(image, band_names):
    """
    Normalises each image band to the range [0, 1] using its 2nd and
    98th percentile values while preserving NaN values.

    Args:
        image (ndarray): Image array with shape (bands, H, W).
        band_names (list): Names of the image bands.

    Returns:
        ndarray: Normalised image array with the same shape as the input.
    """
    normalised_image = np.empty_like(image, dtype=np.float32)

    for i, _ in enumerate(band_names):
        band_data = image[i]
        p2 = np.nanpercentile(band_data, 2)
        p98 = np.nanpercentile(band_data, 98)

        normalised_band = (band_data - p2) / (p98 - p2)
        normalised_band = np.clip(normalised_band, 0, 1)
        normalised_band[np.isnan(band_data)] = np.nan

        normalised_image[i] = normalised_band

    return normalised_image


def extract_and_filter_patches(image, label, patch_size=256, stride=128, nan_threshold=0.5):
    """
    Extracts fixed-size image and label patches and removes patches
    whose proportion of NaN values exceeds the specified threshold.

    Args:
        image (ndarray): Image array with shape (bands, H, W).
        label (ndarray): Corresponding label array with shape (H, W).
        patch_size (int): Height and width of each extracted patch.
        stride (int): Number of pixels between consecutive patch positions.
        nan_threshold (float): Maximum proportion of NaN values allowed in a retained image patch.

    Returns:
        tuple: Retained image patches, label patches, patch positions, and extraction statistics.
    """
    filtered_patches = []
    filtered_labels = []
    positions = []
    total_patches = 0
    discarded_patches = 0

    for i in range(0, image.shape[1] - patch_size + 1, stride):
        for j in range(0, image.shape[2] - patch_size + 1, stride):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            label_patch = label[i:i+patch_size, j:j+patch_size]
            total_patches += 1

            nan_fraction = np.isnan(patch).sum() / patch.size

            if nan_fraction <= nan_threshold:
                filtered_patches.append(patch)
                filtered_labels.append(label_patch)
                positions.append((i, j))
            else:
                discarded_patches += 1

    stats = {
        "total_patches": total_patches,
        "filtered_patches": len(filtered_patches),
        "discarded_patches": discarded_patches
    }
    print(f"Patch size: {patch_size}, Stride: {stride}")
    print(f"Total patches extracted: {stats['total_patches']}")
    print(f"Patches kept: {stats['filtered_patches']}")
    print(f"Patches discarded (NaN > {nan_threshold*100:.0f}%): {stats['discarded_patches']}")

    return filtered_patches, filtered_labels, positions, stats