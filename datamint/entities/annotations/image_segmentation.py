"""Image segmentation annotation entity module for DataMint API.

This module defines the ImageSegmentation class for representing 2D segmentation
annotations in medical images.
"""

from .annotation import Annotation
from datamint.api.dto import AnnotationType
import numpy as np
from PIL import Image
from pydantic import PrivateAttr
import logging

_LOGGER = logging.getLogger(__name__)


class ImageSegmentation(Annotation):
    """
    Image-level (2D) segmentation annotation entity.
    
    Represents a 2D segmentation mask for a single 2d image.
    Supports both binary segmentation (single class) and multi-class 
    semantic segmentation.
    
    This class provides factory methods to create annotations from numpy 
    arrays or PIL Images, which can then be uploaded via AnnotationsApi.
    
    Example:
        >>> # From binary mask
        >>> mask = np.zeros((256, 256), dtype=np.uint8)
        >>> mask[100:150, 100:150] = 1  # lesion region
        >>> img_seg = ImageSegmentation.from_mask(
        ...     mask=mask,
        ...     name='lesion'
        ... )
        >>> 
        >>> # Upload via API
        >>> api.annotations.upload_segmentations(
        ...     resource='resource_id',
        ...     file_path=img_seg.mask,
        ...     name=img_seg.name
        ... )
    """

    _mask: np.ndarray | Image.Image | None = PrivateAttr(default=None)
    _class_name: str | None = PrivateAttr(default=None)

    def __init__(self,
                 name: str | None = None,
                 mask: np.ndarray | Image.Image | None = None,
                 **kwargs):
        """
        Initialize an ImageSegmentation annotation.
        
        Args:
            name: The name/label for this segmentation class
            mask: Optional 2D numpy array or PIL Image containing the segmentation mask
            **kwargs: Additional fields passed to parent Annotation class
        """
        super().__init__(
            identifier=name or "",
            scope='image',
            annotation_type=AnnotationType.SEGMENTATION,
            **kwargs
        )
        
        self._mask = mask
        self._class_name = name

    @classmethod
    def from_mask(cls,
                  mask: np.ndarray | Image.Image,
                  name: str,
                  **kwargs) -> 'ImageSegmentation':
        """
        Create ImageSegmentation from a binary or class mask.
        
        Args:
            mask: 2D numpy array (H x W) with integer labels or binary values,
                or a PIL Image
            name: The name/label for this segmentation
            **kwargs: Additional annotation fields (imported_from, model_id, etc.)
        
        Returns:
            ImageSegmentation instance ready for upload
        
        Raises:
            ValueError: If mask shape is invalid or data types are incorrect
        
        Example:
            >>> mask = np.zeros((512, 512), dtype=np.uint8)
            >>> mask[200:300, 200:300] = 255  # binary mask
            >>> img_seg = ImageSegmentation.from_mask(
            ...     mask=mask,
            ...     name='tumor',
            ... )
        """
        # Convert PIL Image to numpy if needed
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = mask
        
        # Validate mask array
        mask_array = cls._validate_mask_array(mask_array)
        
        instance = cls(
            name=name,
            mask=mask_array,
            **kwargs
        )
        
        return instance

    @staticmethod
    def _validate_mask_array(arr: np.ndarray) -> np.ndarray:
        """
        Validate mask array shape and dtype.
        
        Args:
            arr: Input array to validate
        
        Returns:
            Validated array (possibly with dtype conversion)
        
        Raises:
            ValueError: If array is invalid
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(arr)}")
        
        # Check dimensionality - should be 2D (H x W)
        if arr.ndim != 2:
            raise ValueError(
                f"Mask must be 2D (H x W), got shape {arr.shape}"
            )
        
        # Check dtype - convert floats to int if they're effectively integers
        if np.issubdtype(arr.dtype, np.floating):
            if not np.allclose(arr, arr.astype(int)):
                raise ValueError(
                    "Mask array contains non-integer float values"
                )
            arr = arr.astype(np.uint8)
        elif not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"Mask must have integer dtype, got {arr.dtype}"
            )
        
        # Check for negative values
        if np.any(arr < 0):
            raise ValueError("Mask array contains negative values")
        
        return arr

    @property
    def mask(self) -> np.ndarray | None:
        """
        Get the stored segmentation mask.
        
        Returns:
            2D numpy array or None if not stored
        """
        return self._mask

    @property
    def mask_shape(self) -> tuple[int, int] | None:
        """
        Get the shape of the stored mask.
        
        Returns:
            Shape tuple (H, W) or None if no mask stored
        """
        if self._mask is None:
            return None
        
        if isinstance(self._mask, Image.Image):
            return (self._mask.height, self._mask.width)
        
        return self._mask.shape

    @property
    def class_name(self) -> str | None:
        """
        Get the class name for this segmentation.
        
        Returns:
            Class name string or None
        """
        return self._class_name

    @property
    def name(self) -> str | None:
        """
        Alias for class_name.
        
        Returns:
            Class name string or None
        """
        return self._class_name

    def to_pil_image(self) -> Image.Image | None:
        """
        Convert the mask to a PIL Image.
        
        Returns:
            PIL Image or None if no mask stored
        """
        if self._mask is None:
            return None
        
        if isinstance(self._mask, Image.Image):
            return self._mask
        
        return Image.fromarray(self._mask)

    def get_binary_mask(self, threshold: int = 0) -> np.ndarray | None:
        """
        Get a binary version of the mask.
        
        Args:
            threshold: Values above this threshold are set to 1
        
        Returns:
            Binary numpy array (0s and 1s) or None if no mask stored
        """
        if self._mask is None:
            return None
        
        if isinstance(self._mask, Image.Image):
            mask_array = np.array(self._mask)
        else:
            mask_array = self._mask
        
        return (mask_array > threshold).astype(np.uint8)

    def get_area(self) -> int | None:
        """
        Get the area (number of positive pixels) of the mask.
        
        Returns:
            Number of non-zero pixels or None if no mask stored
        """
        if self._mask is None:
            return None
        
        if isinstance(self._mask, Image.Image):
            mask_array = np.array(self._mask)
        else:
            mask_array = self._mask
        
        return int(np.count_nonzero(mask_array))
