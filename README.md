# MSc Project Source Code

This repository contains the source code for the MSc project on multi-sensor
fusion for land-cover semantic segmentation using Sentinel-1 SAR and
Sentinel-2 optical imagery.

## Code Structure

- `main.ipynb` - Main notebook used to coordinate the experimental pipeline.
- `preprocessing.py` - Data loading, normalisation and patch extraction.
- `splitting.py` - Spatial dataset splitting and class-weight calculation.
- `dataset.py` - PyTorch dataset creation and data augmentation.
- `models.py` - U-Net, Middle Fusion and SegFormer model architectures.
- `losses.py` - Loss-function implementation.
- `train.py` - Model training, validation, checkpointing and early stopping.
- `evaluate.py` - Model evaluation, metrics and visualisation.
- `bootstrap.py` - Paired patch-level bootstrap significance testing.

## Requirements

The code was developed in Python and uses the following main libraries:

- PyTorch
- Segmentation Models PyTorch
- Albumentations
- NumPy
- pandas
- Rasterio
- scikit-learn
- Iterative Stratification
- Google Earth Engine Python API

## Running the Code

1. Install the required Python libraries.
2. Open `main.ipynb` in Google Colab or a compatible Jupyter environment.
3. Ensure that all Python source files are in the same directory as `main.ipynb`.
4. Update the data paths in `main.ipynb` to point to the required input data.
5. Authenticate Google Earth Engine when prompted.
6. Replace `GEE_PROJECT_ID` with an authorised Google Earth Engine project ID.
7. Run the notebook cells in order from top to bottom.

The notebook imports the supporting Python modules and coordinates the complete
experimental pipeline, including data preprocessing, dataset construction,
model training, evaluation and statistical comparison.

GPU acceleration is recommended for model training. The experiments reported
in the dissertation were conducted using an NVIDIA T4 GPU in Google Colab.