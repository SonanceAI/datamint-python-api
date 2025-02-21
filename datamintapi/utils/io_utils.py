import numpy as np
import nibabel as nib
from PIL import Image
from .dicom_utils import load_image_normalized, is_dicom
import pydicom
import os
from typing import Any
import logging
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

_LOGGER = logging.getLogger(__name__)


def read_array_normalized(file_path: str,
                          index: int = None,
                          return_metainfo: bool = False) -> np.ndarray | tuple[np.ndarray, Any]:
    """
    Read an array from a file.

    Args:
        file_path: The path to the file.
        Supported file formats are NIfTI (.nii, .nii.gz), PNG (.png), JPEG (.jpg, .jpeg) and npy (.npy).

    Returns:
        The array read from the file in shape (#frames, C, H, W), if `index=None`,
            or (C, H, W) if `index` is specified.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metainfo = None

    try:

        if is_dicom(file_path):
            ds = pydicom.dcmread(file_path)
            if index is not None:
                imgs = load_image_normalized(ds, index=index)[0]
            else:
                imgs = load_image_normalized(ds)
            # Free up memory
            if hasattr(ds, '_pixel_array'):
                ds._pixel_array = None
            if hasattr(ds, 'PixelData'):
                ds.PixelData = None
            metainfo = ds
        else:
            if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
                imgs = nib.load(file_path).get_fdata()  # shape: (W, H, #frame) or (W, H)
                if imgs.ndim == 2:
                    imgs = imgs.transpose(1, 0)
                    imgs = imgs[np.newaxis, np.newaxis]
                elif imgs.ndim == 3:
                    imgs = imgs.transpose(2, 1, 0)
                    imgs = imgs[:, np.newaxis]
                else:
                    raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim}")
            elif file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                with Image.open(file_path) as pilimg:
                    imgs = np.array(pilimg)
                if imgs.ndim == 2:  # (H, W)
                    imgs = imgs[np.newaxis, np.newaxis]
                elif imgs.ndim == 3:  # (H, W, C)
                    imgs = imgs.transpose(2, 0, 1)[np.newaxis]  # (H, W, C) -> (1, C, H, W)
            elif file_path.endswith('.npy'):
                imgs = np.load(file_path)
                if imgs.ndim != 4:
                    raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim}")
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            if index is not None:
                if len(imgs) > 1:
                    _LOGGER.warning(f"It is inefficient to load all frames from '{file_path}' to access a single frame." +
                                    " Consider converting the file to a format that supports random access (DICOM), or" +
                                    " convert to png/jpeg files or" +
                                    " manually handle all frames at once instead of loading a specific frame.")
                imgs = imgs[index]

        if return_metainfo:
            return imgs, metainfo
        return imgs

    except Exception as e:
        _LOGGER.error(f"Failed to read array from '{file_path}': {e}")
        raise e
