"""Base segmentation annotation module for DataMint API.

Provides shared serialisation helpers and :class:`BaseSegmentationAnnotation`
that is extended by :class:`ImageSegmentation` and :class:`VolumeSegmentation`.
"""

from __future__ import annotations

import base64
import gzip
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, overload

import numpy as np
from PIL import Image
from pydantic import BeforeValidator, PlainSerializer

from .annotation import Annotation
from datamint.types import ImagingData

if TYPE_CHECKING:
    from nibabel.nifti1 import Nifti1Image

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Serialisation helpers (support ndarray, Nifti1Image and PIL Image)
# ---------------------------------------------------------------------------

def _serialize_segmentation_data(
    value: np.ndarray | Image.Image | Any | None,
) -> dict | None:
    """Serialise segmentation data to a JSON-compatible dict.

    Supported types:

    * ``np.ndarray``      – gzip-compressed ``.npy`` bytes, base64-encoded.
    * ``PIL.Image.Image`` – gzip-compressed PNG bytes, base64-encoded.
    * ``Nifti1Image``     – gzip-compressed NIfTI bytes, base64-encoded.
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

    if isinstance(value, Image.Image):
        buf = io.BytesIO()
        value.save(buf, format="PNG")
        compressed = gzip.compress(buf.getvalue())
        return {
            "type": "pil",
            "data": base64.b64encode(compressed).decode("ascii"),
        }

    # Defer Nifti1Image import to avoid hard dependency when not needed
    try:
        from nibabel.nifti1 import Nifti1Image
        if isinstance(value, Nifti1Image):
            compressed = gzip.compress(value.to_bytes())
            return {
                "type": "nifti",
                "data": base64.b64encode(compressed).decode("ascii"),
            }
    except ImportError:
        pass

    raise TypeError(f"Cannot serialise segmentation_data of type {type(value)}")


def _deserialize_segmentation_data(
    value: dict | np.ndarray | Image.Image | Any | None,
) -> np.ndarray | Image.Image | Any | None:
    """Deserialise segmentation data from a JSON-compatible dict or pass-through native types."""
    if value is None:
        return None

    # Already a native type – constructed in code, not from JSON
    if isinstance(value, (np.ndarray, Image.Image)):
        return value

    try:
        from nibabel.nifti1 import Nifti1Image
        if isinstance(value, Nifti1Image):
            return value
    except ImportError:
        pass

    if not isinstance(value, dict):
        raise ValueError(f"Cannot deserialise segmentation_data from {type(value)}")

    raw = gzip.decompress(base64.b64decode(value["data"]))

    dtype = value["type"]
    if dtype == "ndarray":
        return np.load(io.BytesIO(raw))

    if dtype == "pil":
        return Image.open(io.BytesIO(raw))

    if dtype == "nifti":
        from nibabel.nifti1 import Nifti1Image
        return Nifti1Image.from_bytes(raw)

    raise ValueError(f"Unknown segmentation_data type: {dtype!r}")


SegmentationDataType = Annotated[
    np.ndarray | Image.Image | Any | None,
    BeforeValidator(_deserialize_segmentation_data),
    PlainSerializer(_serialize_segmentation_data, return_type=dict | None),
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseSegmentationAnnotation(Annotation):
    """Common base for 2-D and 3-D segmentation annotation entities.

    Provides:

    * A Pydantic-native ``segmentation_data`` field with automatic
      serialisation/deserialisation for ``np.ndarray``, ``PIL.Image.Image``
      and ``nibabel.nifti1.Nifti1Image``.
    * An overridden :meth:`fetch_file_data` that short-circuits the network
      call when ``segmentation_data`` is already populated in memory.
    * Static helpers :meth:`_to_raw_bytes` and :meth:`_from_raw_bytes` for
      subclasses that need to convert to/from ``bytes``.
    """

    segmentation_data: SegmentationDataType = None

    # ------------------------------------------------------------------
    # fetch_file_data
    # ------------------------------------------------------------------

    @overload
    def fetch_file_data(
        self,
        auto_convert: Literal[True] = True,
        save_path: str | None = None,
        use_cache: bool = False,
    ) -> ImagingData: ...

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
    ) -> bytes | ImagingData:
        """Return segmentation file data.

        If :attr:`segmentation_data` is already loaded (e.g. created via a
        factory class-method) the data is returned directly without hitting the
        network.  Otherwise the parent :class:`Annotation` implementation
        downloads the file from the server.

        Args:
            auto_convert: When *True* return the native Python / numpy object;
                when *False* return raw ``bytes`` (NIfTI-encoded for volumes,
                PNG-encoded for 2-D images).
            save_path: Optional path to persist the data on disk.
            use_cache: Whether to use disk-cached data when the API would
                otherwise be called.

        Returns:
            Segmentation data as a native object (when *auto_convert* is
            ``True``) or ``bytes``.
        """
        if self.segmentation_data is not None:
            raw_bytes: bytes | None = None

            if not auto_convert or save_path is not None:
                raw_bytes = self._to_raw_bytes(self.segmentation_data)

                if save_path is not None:
                    Path(save_path).write_bytes(raw_bytes)

            if not auto_convert:
                return raw_bytes  # type: ignore[return-value]

            return self.segmentation_data  # type: ignore[return-value]

        # Fall back to API-based download
        return super().fetch_file_data(
            auto_convert=auto_convert,
            save_path=save_path,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    # Byte conversion helpers (used by subclasses and fetch_file_data)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_raw_bytes(data: np.ndarray | Image.Image | Any) -> bytes:
        """Encode *data* to bytes suitable for file storage.

        * ``np.ndarray``  → NIfTI bytes (wrapped with identity affine).
        * ``PIL.Image``   → PNG bytes.
        * ``Nifti1Image`` → NIfTI bytes.
        """
        if isinstance(data, np.ndarray):
            from nibabel.nifti1 import Nifti1Image
            return Nifti1Image(data, affine=np.eye(4)).to_bytes()

        if isinstance(data, Image.Image):
            buf = io.BytesIO()
            data.save(buf, format="PNG")
            return buf.getvalue()

        try:
            from nibabel.nifti1 import Nifti1Image
            if isinstance(data, Nifti1Image):
                return data.to_bytes()
        except ImportError:
            pass

        raise TypeError(f"Cannot convert {type(data)} to bytes")

    @staticmethod
    def _from_raw_bytes(raw: bytes, as_pil: bool = False) -> np.ndarray | Image.Image | Any:
        """Decode *raw* bytes back to a native object.

        Args:
            raw: Bytes to decode.
            as_pil: If ``True``, attempt to decode as a PIL Image.

        Returns:
            Decoded object.
        """
        if as_pil:
            return Image.open(io.BytesIO(raw))

        try:
            from nibabel.nifti1 import Nifti1Image
            return Nifti1Image.from_bytes(raw)
        except Exception:
            pass

        raise ValueError("Could not decode bytes as NIfTI or PIL Image")
