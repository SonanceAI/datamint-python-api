import numpy as np
import nibabel as nib
from PIL import Image
from .dicom_utils import load_image_normalized, is_dicom
import pydicom
import os
from typing import Any
import logging
from PIL import ImageFile
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

_LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')
NII_EXTS = ('.nii', '.nii.gz')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


def read_video(file_path: str, index: int = None) -> np.ndarray:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {file_path}")
    try:
        if index is None:
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB and transpose to (C, H, W) format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.transpose(2, 0, 1)
                frames.append(frame)
            imgs = np.array(frames)  # shape: (#frames, C, H, W)
        else:
            while index > 0:
                cap.grab()
                index -= 1
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {index} from video file: {file_path}")
            # Convert BGR to RGB and transpose to (C, H, W) format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs = frame.transpose(2, 0, 1)
    finally:
        cap.release()

    if imgs is None or len(imgs) == 0:
        raise ValueError(f"No frames found in video file: {file_path}")

    return imgs


def read_nifti(file_path: str) -> np.ndarray:
    imgs = nib.load(file_path).get_fdata()  # shape: (W, H, #frame) or (W, H)
    if imgs.ndim == 2:
        imgs = imgs.transpose(1, 0)
        imgs = imgs[np.newaxis, np.newaxis]
    elif imgs.ndim == 3:
        imgs = imgs.transpose(2, 1, 0)
        imgs = imgs[:, np.newaxis]
    else:
        raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim}")

    return imgs


def read_image(file_path: str) -> np.ndarray:
    with Image.open(file_path) as pilimg:
        imgs = np.array(pilimg)
    if imgs.ndim == 2:  # (H, W)
        imgs = imgs[np.newaxis, np.newaxis]
    elif imgs.ndim == 3:  # (H, W, C)
        imgs = imgs.transpose(2, 0, 1)[np.newaxis]  # (H, W, C) -> (1, C, H, W)

    return imgs


def read_array_normalized(file_path: str,
                          index: int = None,
                          return_metainfo: bool = False,
                          use_magic=False) -> np.ndarray | tuple[np.ndarray, Any]:
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
            if use_magic:
                import magic  # it is important to import here because magic has an OS lib dependency.
                mime_type = magic.from_file(file_path, mime=True)
            else:
                mime_type = ""

            if mime_type.startswith('video/') or file_path.endswith(VIDEO_EXTS):
                imgs = read_video(file_path, index)
            else:
                if mime_type == 'image/x.nifti' or file_path.endswith(NII_EXTS):
                    imgs = read_nifti(file_path)
                elif mime_type.startswith('image/') or file_path.endswith(IMAGE_EXTS):
                    imgs = read_image(file_path)
                elif file_path.endswith('.npy') or mime_type == 'application/x-numpy-data':
                    imgs = np.load(file_path)
                    if imgs.ndim != 4:
                        raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim}")
                else:
                    raise ValueError(f"Unsupported file format '{mime_type}' of '{file_path}'")

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
