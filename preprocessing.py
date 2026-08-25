import numpy as np
import rasterio


def file_upload(file_path):
    """
    Loads a GeoTIFF into a NumPy array, converting the NoData
    sentinel value back to NaN.

    Args:
        file_path (str): path to the GeoTIFF file.

    Returns:
        tuple: (image array, rasterio profile dict).
    """
    with rasterio.open(file_path) as src:
        image = src.read().astype('float32')
        profile = src.profile.copy()
        profile.update(dtype='float32', nodata=np.nan)

        image[image == -9999] = np.nan

    return image, profile


def percentile_normalise(image, band_names):
    """
    Normalizes each band of an image to [0, 1] using 2nd-98th
    percentile stretching, preserving NaN values.

    Args:
        image (ndarray): image array, shape (bands, H, W).
        band_names (list): names of each band, for reference.

    Returns:
        ndarray: normalized image, same shape as input.
    """
    normalised_image = np.empty_like(image, dtype=np.float32)

    for i, band in enumerate(band_names):
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
    Extracts fixed-size patches from a raster and its corresponding
    label array, discarding patches whose NaN proportion exceeds
    a given threshold.

    Args:
        image (ndarray): image array, shape (bands, H, W).
        label (ndarray): label array, shape (H, W).
        patch_size (int): height/width of each extracted patch.
        stride (int): step size between adjacent patch positions.
        nan_threshold (float): maximum allowed fraction of NaN pixels
            per patch before it is discarded.

    Returns:
        tuple: (list of retained image patches, list of retained label
                patches, list of (row, col) positions, dict of extraction
                statistics).
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