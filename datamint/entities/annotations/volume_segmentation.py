"""Volume segmentation annotation entity module for DataMint API.

This module defines the VolumeSegmentation class for representing 3D segmentation
annotations in medical imaging volumes.
"""

from .annotation import Annotation
from datamint.api.dto import AnnotationType
import numpy as np
from nibabel.nifti1 import Nifti1Image
from pydantic import BeforeValidator, PlainSerializer

import base64
import gzip
import io
import logging
from typing import Annotated, Literal, overload

from datamint.types import ImagingData

_LOGGER = logging.getLogger(__name__)


def _serialize_segmentation_data(value: np.ndarray | Nifti1Image | None) -> dict | None:
    """Serialize segmentation data to a JSON-compatible dict.

    For np.ndarray: gzip-compressed .npy bytes, base64-encoded.
    For Nifti1Image: gzip-compressed NIfTI bytes, base64-encoded.
    """
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, value)
        compressed = gzip.compress(buf.getvalue())
        return {
            "type": "ndarray",
            "data": base64.b64encode(compressed).decode("ascii"),
        }
    elif isinstance(value, Nifti1Image):
        buf = io.BytesIO()
        # Nifti1Image.to_bytes() returns bytes of the full NIfTI file
        nifti_bytes = value.to_bytes()
        compressed = gzip.compress(nifti_bytes)
        return {
            "type": "nifti",
            "data": base64.b64encode(compressed).decode("ascii"),
        }
    else:
        raise TypeError(f"Cannot serialize segmentation_data of type {type(value)}")


def _deserialize_segmentation_data(value: dict | np.ndarray | Nifti1Image | None) -> np.ndarray | Nifti1Image | None:
    """Deserialize segmentation data from a JSON-compatible dict or pass through native types."""
    if value is None:
        return None

    # Already the correct type (e.g. constructed in code, not from JSON)
    if isinstance(value, (np.ndarray, Nifti1Image)):
        return value

    if not isinstance(value, dict):
        raise ValueError(f"Cannot deserialize segmentation_data from {type(value)}")

    raw = gzip.decompress(base64.b64decode(value["data"]))

    if value["type"] == "ndarray":
        buf = io.BytesIO(raw)
        return np.load(buf)
    elif value["type"] == "nifti":
        from nibabel.nifti1 import Nifti1Image as _Nifti1Image
        fh = io.BytesIO(raw)
        return _Nifti1Image.from_bytes(fh.read())
    else:
        raise ValueError(f"Unknown segmentation_data type: {value['type']}")


SegmentationDataType = Annotated[
    np.ndarray | Nifti1Image | None,
    BeforeValidator(_deserialize_segmentation_data),
    PlainSerializer(_serialize_segmentation_data, return_type=dict | None),
]



class VolumeSegmentation(Annotation):
    """
    Volume-level segmentation annotation entity.
    
    Represents a 3D segmentation mask for medical imaging volumes.
    Supports both semantic segmentation (class per voxel) and instance 
    segmentation (unique ID per object).
    
    This class provides factory methods to create annotations from numpy 
    arrays or NIfTI images, which can then be uploaded via AnnotationsApi.
    
    Example:
        >>> # From semantic segmentation
        >>> seg_data = np.array([...])  # Shape: (H, W, D)
        >>> class_map = {1: 'tumor', 2: 'edema'}
        >>> vol_seg = VolumeSegmentation.from_semantic_segmentation(
        ...     segmentation=seg_data,
        ...     class_map=class_map
        ... )
        >>> 
        >>> # Upload via API
        >>> api.annotations.upload_segmentations(
        ...     resource='resource_id',
        ...     file_path=vol_seg.segmentation_data,
        ...     name=vol_seg.class_map
        ... )
    """

    segmentation_data: SegmentationDataType = None
    class_map: dict[int, str] | None = None

    
    def __init__(self,
                 **kwargs):
        """
        Initialize a VolumeSegmentation annotation.
        
        Args:
            **kwargs: Additional fields passed to parent Annotation class
        """
        kwargs['scope'] = 'image'
        kwargs['annotation_type'] = AnnotationType.SEGMENTATION

        kwargs.setdefault('identifier', '')
        super().__init__(**kwargs)
        
    @classmethod
    def from_semantic_segmentation(cls,
                                   segmentation: np.ndarray | Nifti1Image,
                                   class_map: dict[int, str] | str,
                                   **kwargs) -> 'VolumeSegmentation':
        """
        Create VolumeSegmentation from semantic segmentation data.
        
        Semantic segmentation: each voxel has a single integer label 
        corresponding to its class.
        
        Args:
            segmentation: 3D numpy array (H x W x D) or NIfTI image with 
                integer labels representing classes
            class_map: Mapping from label integers to class names, or a 
                single class name for binary segmentation (background=0, class=1)
            **kwargs: Additional annotation fields (imported_from, model_id, etc.)
        
        Returns:
            VolumeSegmentation instance ready for upload
        
        Raises:
            ValueError: If segmentation shape is invalid, class_map is incomplete,
                or data types are incorrect
        
        Example:
            >>> seg = np.zeros((256, 256, 128), dtype=np.int32)
            >>> seg[100:150, 100:150, 50:75] = 1  # tumor region
            >>> vol_seg = VolumeSegmentation.from_semantic_segmentation(
            ...     segmentation=seg,
            ...     class_map={1: 'tumor'}, # or just ``class_map='tumor'``
            ... )
        """
        # Step 1: Convert Nifti1Image to numpy if needed
        if isinstance(segmentation, Nifti1Image):
            seg_array = segmentation.get_fdata().astype(np.int32)
        else:
            seg_array = segmentation
        
        # Step 2: Validate segmentation array
        seg_array = cls._validate_segmentation_array(seg_array)
        
        # Step 3: Standardize class_map to dict[int, str]
        standardized_class_map = cls._standardize_class_map(class_map, seg_array)
        
        instance = cls(**kwargs)
        
        instance.segmentation_data = segmentation
        instance.class_map = standardized_class_map
        
        return instance

    @staticmethod
    def _validate_segmentation_array(arr: np.ndarray) -> np.ndarray:
        """
        Validate segmentation array shape and dtype.
        
        Args:
            arr: Input array to validate
        
        Returns:
            Validated array (possibly with dtype conversion)
        
        Raises:
            ValueError: If array is invalid
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(arr)}")
        
        # Check dimensionality
        if arr.ndim != 3:
            raise ValueError(
                f"Segmentation must be 3D (H x W x D), got shape {arr.shape}"
            )
        
        # Check dtype
        if not np.issubdtype(arr.dtype, np.integer):
            # Try to convert to int
            if np.issubdtype(arr.dtype, np.floating):
                # Check if values are effectively integers
                if not np.allclose(arr, arr.astype(int)):
                    raise ValueError(
                        "Segmentation array contains non-integer float values"
                    )
                arr = arr.astype(np.int32)
            else:
                raise ValueError(
                    f"Segmentation must have integer dtype, got {arr.dtype}"
                )
        
        # Check for negative values
        if np.any(arr < 0):
            raise ValueError("Segmentation array contains negative values")
        
        return arr

    @staticmethod
    def _standardize_class_map(
        class_map: dict[int, str] | str,
        segmentation: np.ndarray
    ) -> dict[int, str]:
        """
        Convert class_map to standard dict[int, str] format.
        
        Args:
            class_map: Either a dict or a single class name for binary seg
            segmentation: The segmentation array to infer labels from
        
        Returns:
            Standardized dictionary mapping labels to class names
        
        Raises:
            ValueError: If class_map format is invalid
        """
        if isinstance(class_map, str):
            # Binary segmentation: assume label 1 = class_map, 0 = background
            unique_labels = np.unique(segmentation)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude 0
            
            if len(unique_labels) != 1:
                raise ValueError(
                    f"Single class name provided but segmentation has "
                    f"{len(unique_labels)} non-zero labels: {unique_labels.tolist()}"
                )
            
            return {int(unique_labels[0]): class_map}
        
        elif isinstance(class_map, dict):
            # Validate all keys are integers, all values are strings
            standardized = {}
            for k, v in class_map.items():
                if not isinstance(k, (int, np.integer)):
                    raise ValueError(f"class_map key must be integer, got {type(k)}")
                if not isinstance(v, str):
                    raise ValueError(f"class_map value must be string, got {type(v)}")
                standardized[int(k)] = v
            
            return standardized
        
        else:
            raise ValueError(
                f"class_map must be dict[int, str] or str, got {type(class_map)}"
            )



    @property
    def volume_shape(self) -> tuple[int, int, int] | None:
        """
        Get the shape of the stored segmentation volume.
        
        Returns:
            Shape tuple (H, W, D) or None if no data stored
        """
        if self.segmentation_data is None:
            return None
        
        if isinstance(self.segmentation_data, Nifti1Image):
            shape = self.segmentation_data.shape
            return (shape[0], shape[1], shape[2])
        else:
            return self.segmentation_data.shape

    @property
    def class_names(self) -> list[str] | None:
        """
        Get list of class names from stored class_map.
        
        Returns:
            List of class names or None if no class_map stored
        """
        if self.class_map is None:
            return None
        return sorted(self.class_map.values())

    @property
    def num_classes(self) -> int | None:
        """
        Get number of classes in this segmentation.
        
        Returns:
            Number of classes or None if no class_map stored
        """
        if self.class_map is None:
            return None
        return len(self.class_map)

    @overload
    def fetch_file_data(
        self,
        auto_convert: Literal[True] = True,
        save_path: str | None = None,
        use_cache: bool = False,
    ) -> 'ImagingData': ...

    @overload
    def fetch_file_data(
        self,
        auto_convert: Literal[False],
        save_path: str | None = None,
        use_cache: bool = False,
    ) -> bytes: ...

    def fetch_file_data(
        self,
        auto_convert: bool = True,
        save_path: str | None = None,
        use_cache: bool = False,
    ) -> 'bytes | ImagingData':
        """Get the file data for this volume segmentation annotation.

        If ``segmentation_data`` is already populated (e.g. created via
        :meth:`from_semantic_segmentation`), it is returned directly without
        making an API call.  Otherwise the parent implementation downloads
        the data from the server.

        Args:
            auto_convert: If True, return the data as a numpy array /
                Nifti1Image.  If False, return raw bytes.
            save_path: Optional path to save the file locally.
            use_cache: If True, uses cached data when available and valid.

        Returns:
            Segmentation data as ``np.ndarray`` / ``Nifti1Image`` (when
            *auto_convert* is True) or raw ``bytes``.
        """
        if self.segmentation_data is not None:
            if not auto_convert:
                # Convert in-memory data to NIfTI bytes
                if isinstance(self.segmentation_data, Nifti1Image):
                    raw_bytes = self.segmentation_data.to_bytes()
                else:
                    # Wrap ndarray in a minimal NIfTI image to produce bytes
                    nifti_img = Nifti1Image(self.segmentation_data, affine=np.eye(4))
                    raw_bytes = nifti_img.to_bytes()

                if save_path is not None:
                    from pathlib import Path
                    Path(save_path).write_bytes(raw_bytes)

                return raw_bytes

            # auto_convert=True: return the in-memory object directly
            if save_path is not None:
                from pathlib import Path
                if isinstance(self.segmentation_data, Nifti1Image):
                    raw_bytes = self.segmentation_data.to_bytes()
                else:
                    nifti_img = Nifti1Image(self.segmentation_data, affine=np.eye(4))
                    raw_bytes = nifti_img.to_bytes()
                Path(save_path).write_bytes(raw_bytes)

            return self.segmentation_data

        # No local data – fall back to the API-based download
        return super().fetch_file_data(
            auto_convert=auto_convert,
            save_path=save_path,
            use_cache=use_cache,
        )

