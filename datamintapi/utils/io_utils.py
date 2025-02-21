import numpy as np
import nibabel as nib
from PIL import Image
from .dicom_utils import load_image_normalized
import pydicom
import os


def read_array_normalized(file_path: str) -> np.ndarray:
    """
    Read an array from a file.

    Args:
        file_path: The path to the file.
        Supported file formats are NIfTI (.nii, .nii.gz), PNG (.png), JPEG (.jpg, .jpeg) and npy (.npy).

    Returns:
        The array read from the file in shape (#frames, C, H, W)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        imgs = nib.load(file_path).get_fdata()  # shape: (W, H, #frame) or (W, H)
        if imgs.ndim != 3 and imgs.ndim != 2:
            raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim}")
        imgs = imgs.transpose(2, 1, 0) if imgs.ndim == 3 else imgs.transpose(1, 0)
        return imgs
    elif file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
        with Image.open(file_path) as pilimg:
            imgs = np.array(pilimg)
        if imgs.ndim == 2:
            imgs = imgs[np.newaxis, np.newaxis]
        elif imgs.ndim == 3:
            imgs = imgs[:, np.newaxis]
        return imgs
    elif file_path.endswith('.npy'):
        imgs = np.load(file_path)
        if imgs.ndim != 4:
            raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim}")
    elif file_path.endswith('.dcm'):
        return load_image_normalized(pydicom.dcmread(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
