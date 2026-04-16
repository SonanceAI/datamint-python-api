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
from typing import Annotated, Any, Literal, overload

import numpy as np
from PIL import Image
from pydantic import BeforeValidator, PlainSerializer, Field

from .annotation import Annotation
from datamint.types import ImagingData
from nibabel.nifti1 import Nifti1Image

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Serialisation helpers (support ndarray, Nifti1Image and PIL Image)
# ---------------------------------------------------------------------------

# Legacy type string → canonical MIME type
_LEGACY_TYPE_TO_MIMETYPE: dict[str, str] = {
    "ndarray": "application/x-numpy",
    "pil": "image/png",
    "nifti": "application/nifti",
}


def _serialize_segmentation_data(
    value: np.ndarray | Image.Image | Any | None,
    serialize_as_mimetype: str | None = 'application/nifti',
) -> dict | None:
    """Serialise segmentation data to a JSON-compatible dict.

    Supported types and their output MIME types:

    * ``np.ndarray``      → ``application/x-numpy``  (gzip-compressed ``.npy``)
    * ``PIL.Image.Image`` → ``image/png``             (uncompressed PNG)
    * ``Nifti1Image``     → ``application/nifti``  (gzip-compressed NIfTI)

    Args:
        value: The segmentation data to serialise.
        serialize_as_mimetype: When provided, force encoding to the given MIME
            type regardless of the Python type of *value*.  Supported values
            are ``"image/png"`` and ``"application/nifti"``.

    Returns:
        A JSON-compatible dict with keys ``"mimetype"``, ``"data"`` (base64)
        and optionally ``"compressed": true`` when gzip compression was
        applied, or ``None`` when *value* is ``None``.
    """
    if value is None:
        return None

    # ------------------------------------------------------------------ #
    # Forced encoding via serialize_as_mimetype                           #
    # ------------------------------------------------------------------ #
    # Determine the original Python type before any forced conversion
    if isinstance(value, np.ndarray):
        original_type = "ndarray"
    elif isinstance(value, Image.Image):
        original_type = "pil"
    else:
        original_type = "nifti"

    if serialize_as_mimetype == "image/png":
        if isinstance(value, np.ndarray):
            value = Image.fromarray(value)
        elif isinstance(value, Nifti1Image):
            value = Image.fromarray(np.squeeze(np.asarray(value.dataobj)))
        elif not isinstance(value, Image.Image):
            raise TypeError(
                f"Cannot serialise {type(value)} as image/png; "
                "expected np.ndarray or PIL.Image.Image"
            )
        buf = io.BytesIO()
        value.save(buf, format="PNG")
        return {
            "mimetype": "image/png",
            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            "compressed": False,
            "type": original_type,
        }

    if serialize_as_mimetype == "application/nifti":
        if isinstance(value, np.ndarray):
            value = Nifti1Image(value, affine=np.eye(4))
        elif isinstance(value, Image.Image):
            value = Nifti1Image(np.array(value), affine=np.eye(4))
        elif not isinstance(value, Nifti1Image):
            raise TypeError(
                f"Cannot serialise {type(value)} as application/nifti; "
                "expected np.ndarray, PIL.Image.Image or Nifti1Image"
            )

        return {
            "mimetype": "application/nifti",
            "data": base64.b64encode(gzip.compress(value.to_bytes())).decode("ascii"),
            "compressed": True,
            "type": original_type,
        }

    if serialize_as_mimetype is not None and serialize_as_mimetype != 'auto':
        raise ValueError(f"Unsupported serialize_as_mimetype: {serialize_as_mimetype!r}")

    # ------------------------------------------------------------------ #
    # Auto-detect encoding from Python type                               #
    # ------------------------------------------------------------------ #
    if isinstance(value, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, value)
        return {
            "mimetype": "application/x-numpy",
            "data": base64.b64encode(gzip.compress(buf.getvalue())).decode("ascii"),
            "compressed": True,
            "type": "ndarray",
        }

    if isinstance(value, Image.Image):
        buf = io.BytesIO()
        value.save(buf, format="PNG")
        # PNG is already an efficient binary format – no gzip on top
        return {
            "mimetype": "image/png",
            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            "compressed": False,
            "type": "pil",
        }

    return {
        "mimetype": "application/nifti",
        "data": base64.b64encode(gzip.compress(value.to_bytes())).decode("ascii"),
        "compressed": True,
        "type": "nifti",
    }


def _deserialize_segmentation_data(
    value: dict | np.ndarray | Image.Image | Any | list | None,
) -> np.ndarray | Image.Image | Any | None:
    """Deserialise segmentation data from a JSON-compatible dict or pass-through native types.

    Supports both the current format (``"mimetype"`` key) and the legacy
    format (``"type"`` key) for backward compatibility.
    """
    if value is None:
        return None

    # Already a native type – constructed in code, not from JSON
    if isinstance(value, (np.ndarray, Image.Image, Nifti1Image)):
        return value

    if isinstance(value, list):
        return np.array(value)

    if not isinstance(value, dict):
        raise ValueError(f"Cannot deserialise segmentation_data from {type(value)}")

    # Resolve MIME type: prefer "mimetype", fall back to legacy "type" key
    if "mimetype" in value:
        mimetype = value["mimetype"]
        is_legacy = False
    elif "type" in value:
        legacy_type = value["type"]
        mimetype = _LEGACY_TYPE_TO_MIMETYPE.get(legacy_type)
        if mimetype is None:
            raise ValueError(f"Unknown legacy segmentation_data type: {legacy_type!r}")
        is_legacy = True
    else:
        raise ValueError("segmentation_data dict must contain 'mimetype' or 'type' key")

    raw_bytes = base64.b64decode(value["data"])

    # Decompress when explicitly flagged or when reading the legacy format
    # (which always gzip-compressed every payload, including PIL images).
    if value.get("compressed", False) or is_legacy:
        raw_bytes = gzip.decompress(raw_bytes)

    # Decode based on mimetype
    if mimetype == "application/x-numpy":
        decoded = np.load(io.BytesIO(raw_bytes))
    elif mimetype == "image/png":
        decoded = Image.open(io.BytesIO(raw_bytes))
    elif mimetype == "application/nifti":
        decoded = Nifti1Image.from_bytes(raw_bytes)
    else:
        raise ValueError(f"Unknown segmentation_data mimetype: {mimetype!r}")

    # Convert to the original Python type recorded during serialisation
    target_type = value.get("type")
    if target_type is None or is_legacy:
        return decoded

    # Already the right type
    if (
        (target_type == "ndarray" and isinstance(decoded, np.ndarray))
        or (target_type == "pil" and isinstance(decoded, Image.Image))
    ):
        return decoded

    if target_type == "nifti" and isinstance(decoded, Nifti1Image):
        return decoded

    if target_type == "ndarray":
        if isinstance(decoded, Image.Image):
            return np.array(decoded)
        if isinstance(decoded, Nifti1Image):
            return np.asarray(decoded.dataobj)

    if target_type == "pil":
        if isinstance(decoded, np.ndarray):
            return Image.fromarray(decoded)
        if isinstance(decoded, Nifti1Image):
            return Image.fromarray(np.squeeze(np.asarray(decoded.dataobj)))

    if target_type == "nifti":
        if isinstance(decoded, np.ndarray):
            return Nifti1Image(decoded, affine=np.eye(4))
        if isinstance(decoded, Image.Image):
            return Nifti1Image(np.array(decoded), affine=np.eye(4))

    _LOGGER.warning(
        "Could not convert decoded type %s to target type %r; returning as-is.",
        type(decoded).__name__,
        target_type,
    )
    return decoded


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

    segmentation_data: Annotated[SegmentationDataType, Field(alias='mask')] = None

    # ------------------------------------------------------------------
    # fetch_file_data
    # ------------------------------------------------------------------

    def __init__(self, 
                 segmentation_data: np.ndarray | Image.Image | None = None,
                 mask: np.ndarray | Image.Image | None = None,
                 **kwargs
                 ) -> None:
        if mask is not None:
            if segmentation_data is not None:
                raise ValueError("Cannot specify both 'segmentation_data' and 'mask'")
            segmentation_data = mask
        super().__init__(segmentation_data=segmentation_data, **kwargs)

    def _to_create_dto(self):
        raise ValueError(
            'Segmentation annotations require file upload. '
            'Use upload_segmentations or upload_volume_segmentation instead.'
        )

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


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def mask(self) -> np.ndarray | Image.Image | Any | None:
        """Alias for :attr:`segmentation_data`."""
        return self.segmentation_data

    @property
    def mask_shape(self) -> tuple[int, ...] | None:
        """
        Shape of the stored mask.
        """
        data = self.segmentation_data
        if data is None:
            return None
        if isinstance(data, Image.Image):
            return (data.height, data.width)
        return data.shape