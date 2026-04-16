"""Image segmentation annotation entity module for DataMint API.

This module defines the ImageSegmentation class for representing 2D binary
segmentation annotations in medical images.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .types import AnnotationType
from .base_segmentation import BaseSegmentationAnnotation

_LOGGER = logging.getLogger(__name__)


class ImageSegmentation(BaseSegmentationAnnotation):
    """
    Image-level (2D) binary segmentation annotation entity.

    Represents a binary segmentation mask (foreground / background) for a
    single image.  The stored mask always contains only ``0`` and ``1``
    values; any non-zero input pixel is normalised to ``1``.

    :attr:`segmentation_data` stores the mask as a ``np.ndarray`` with
    ``dtype=uint8`` and is automatically serialised/deserialised when the
    annotation is persisted or loaded.

    The annotation name (the class label) is stored in the inherited
    ``identifier`` field and accessible via the :attr:`name` property.

    Example:
        >>> mask = np.zeros((256, 256), dtype=np.uint8)
        >>> mask[100:150, 100:150] = 1  # foreground region
        >>> img_seg = ImageSegmentation.from_mask(mask=mask, name='lesion')
        >>> img_seg.name
        'lesion'
    """

    def __init__(
        self,
        segmentation_data: np.ndarray | Image.Image | None = None,
        **kwargs,
    ) -> None:
        """Initialize an ImageSegmentation annotation.  
        The segmentation data is validated and normalised to a binary mask.

        Args:
            segmentation_data: 2D binary mask as a numpy array or PIL Image.
            **kwargs: Additional fields passed to parent Annotation class
        """
        kwargs['scope'] = 'image'
        kwargs['annotation_type'] = AnnotationType.SEGMENTATION

        if isinstance(segmentation_data, np.ndarray):
            segmentation_data = ImageSegmentation._validate_mask_array(segmentation_data)
        super().__init__(segmentation_data=segmentation_data, **kwargs)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_mask_array(arr: np.ndarray) -> np.ndarray:
        """
        Validate and normalise a binary mask array.

        * Must be 2D.
        * Must have integer or float dtype (floats are accepted only when they
          represent integer values).
        * No negative values.
        * Non-zero values are normalised to ``1`` (``dtype=uint8``).

        Args:
            arr: Input array to validate.

        Returns:
            Validated binary ``uint8`` array with values in ``{0, 1}``.

        Raises:
            ValueError: If the array is invalid.
        """
        if arr.ndim != 2:
            raise ValueError(
                f"Mask must be 2D (H × W), got shape {arr.shape}"
            )

        if np.issubdtype(arr.dtype, np.floating):
            if not np.allclose(arr, arr.astype(int)):
                raise ValueError("Mask array contains non-integer float values")
            arr = arr.astype(np.int32)
        elif not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"Mask must have integer dtype, got {arr.dtype}")

        if np.any(arr < 0):
            raise ValueError("Mask array contains negative values")

        # Normalise to strict binary {0, 1}
        return (arr > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_pil_image(self) -> Image.Image | None:
        """
        Convert the mask to a PIL ``Image``.

        Returns:
            :class:`PIL.Image.Image` or ``None`` if no mask is stored.
        """
        data = self.segmentation_data
        if data is None:
            return None
        if isinstance(data, Image.Image):
            return data
        return Image.fromarray(data)

    def get_area(self) -> int | None:
        """
        Return the number of foreground (non-zero) pixels in the mask.

        Returns:
            Foreground pixel count or ``None`` if no mask is stored.
        """
        data = self.segmentation_data
        if data is None:
            return None
        arr = np.array(data) if isinstance(data, Image.Image) else data
        return int(np.count_nonzero(arr))
