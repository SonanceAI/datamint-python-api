from nibabel.nifti1 import Nifti1Header, Nifti1Image
from collections.abc import Mapping
import numpy as np


def _get_metadata_value(metadata: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _coerce_metadata_int(value: object, field_name: str) -> int:
    if not isinstance(value, (int, float, np.integer, np.floating, str, bytes, bytearray)):
        raise ValueError(f"metadata '{field_name}' must be an integer-compatible value")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata '{field_name}' must be an integer-compatible value") from exc


def _coerce_metadata_float(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating, str, bytes, bytearray)):
        raise ValueError(f"metadata '{field_name}' must be a float-compatible value")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata '{field_name}' must be a float-compatible value") from exc


def _get_nifti_shape_from_metadata(metadata: Mapping[str, object]) -> tuple[int, ...]:
    dims_value = _get_metadata_value(metadata, 'dims', 'dim')
    if dims_value is None:
        raise ValueError("metadata must define 'dims' or 'dim' to reconstruct a NIfTI image")

    dims = np.asarray(dims_value, dtype=np.int64)
    if dims.ndim != 1:
        raise ValueError("metadata 'dims' must be a one-dimensional sequence")
    if dims.size < 2:
        raise ValueError("metadata 'dims' must include ndim plus at least one axis length")

    ndim = int(dims[0])
    if ndim < 1 or ndim > 7:
        raise ValueError(f"metadata 'dims' declares an invalid ndim: {ndim}")
    if dims.size < ndim + 1:
        raise ValueError(
            f"metadata 'dims' does not contain enough axis lengths for ndim={ndim}")

    shape = tuple(int(length) for length in dims[1:ndim + 1])
    if any(length <= 0 for length in shape):
        raise ValueError(f"metadata 'dims' contains non-positive axis lengths: {shape}")
    return shape


def _get_nifti_zooms_from_metadata(metadata: Mapping[str, object], ndim: int) -> tuple[float, ...] | None:
    pix_dims_value = _get_metadata_value(metadata, 'pixDims', 'pixdim')
    if pix_dims_value is None:
        return None

    pix_dims = np.asarray(pix_dims_value, dtype=np.float64)
    if pix_dims.ndim != 1:
        raise ValueError("metadata 'pixDims' must be a one-dimensional sequence")
    if pix_dims.size < ndim + 1:
        raise ValueError(
            f"metadata 'pixDims' does not contain enough zoom values for ndim={ndim}")

    return tuple(
        1.0 if float(zoom) <= 0.0 else float(zoom)
        for zoom in pix_dims[1:ndim + 1]
    )


def _get_nifti_affine_from_metadata(metadata: Mapping[str, object]) -> np.ndarray:
    affine_value = _get_metadata_value(metadata, 'affine')
    if affine_value is None:
        return np.eye(4, dtype=np.float64)

    affine = np.asarray(affine_value, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"metadata 'affine' must have shape (4, 4), got {affine.shape}")
    return affine


def _build_nifti_header_from_metadata(metadata: Mapping[str, object],
                                      shape: tuple[int, ...],
                                      affine: np.ndarray,
                                      data_dtype: np.dtype | None = None) -> Nifti1Header:
    header = Nifti1Header()
    header.set_data_shape(shape)

    datatype = _get_metadata_value(metadata, 'datatype')
    if datatype is not None:
        header.set_data_dtype(_coerce_metadata_int(datatype, 'datatype'))
    elif data_dtype is not None:
        header.set_data_dtype(data_dtype)

    zooms = _get_nifti_zooms_from_metadata(metadata, len(shape))
    if zooms is not None:
        header.set_zooms(zooms)

    qform_code = _get_metadata_value(metadata, 'qformCode', 'qform_code')
    sform_code = _get_metadata_value(metadata, 'sformCode', 'sform_code')
    header.set_qform(
        affine,
        code=None if qform_code is None else _coerce_metadata_int(qform_code, 'qformCode'),
    )
    header.set_sform(
        affine,
        code=None if sform_code is None else _coerce_metadata_int(sform_code, 'sformCode'),
    )

    vox_offset = _get_metadata_value(metadata, 'voxOffset', 'vox_offset')
    if vox_offset is not None:
        header['vox_offset'] = _coerce_metadata_float(vox_offset, 'voxOffset')

    description = _get_metadata_value(metadata, 'description', 'descrip')
    if description is not None:
        header['descrip'] = str(description).encode('utf-8')[:80]

    return header


def metadata_to_nifti_obj(metadata: Mapping[str, object],
                          dataobj: np.ndarray | None = None,
                          *,
                          fill_value: int | float = 0) -> Nifti1Image:
    """Construct a ``Nifti1Image`` from a metadata mapping and optional data.

    nibabel provides the building blocks for this via ``Nifti1Header`` and
    ``Nifti1Image``, but it does not expose a high-level helper for the JSON-like
    metadata dictionaries commonly returned by JavaScript viewers or APIs.

    Metadata alone cannot recover voxel intensities. When ``dataobj`` is not
    provided, this function allocates a zero-filled array using the shape and
    datatype declared in ``metadata``.

    Args:
        metadata: Mapping containing NIfTI metadata. The helper understands the
            common camelCase keys from viewer metadata (for example ``dims``,
            ``pixDims``, ``qformCode``, ``sformCode``, ``voxOffset``) as well as
            the canonical nibabel header names ``dim``, ``pixdim``, ``descrip``,
            and ``vox_offset``.
        dataobj: Optional voxel data. When provided, its shape must match the
            shape declared in ``metadata``.
        fill_value: Scalar used when ``dataobj`` is omitted.

    Returns:
        Nifti1Image: A nibabel image with header fields populated from the
        metadata mapping.
    """
    shape = _get_nifti_shape_from_metadata(metadata)
    affine = _get_nifti_affine_from_metadata(metadata)

    data_array = None if dataobj is None else np.asarray(dataobj)
    if data_array is not None and tuple(data_array.shape) != shape:
        raise ValueError(
            f"dataobj shape {tuple(data_array.shape)} does not match metadata shape {shape}")

    header = _build_nifti_header_from_metadata(
        metadata,
        shape,
        affine,
        data_dtype=None if data_array is None else data_array.dtype,
    )
    target_dtype = header.get_data_dtype()

    if data_array is None:
        data_array = np.full(shape, fill_value, dtype=target_dtype)
    else:
        data_array = data_array.astype(target_dtype, copy=False)

    image = Nifti1Image(data_array, affine, header=header)

    qform_code = _get_metadata_value(metadata, 'qformCode', 'qform_code')
    sform_code = _get_metadata_value(metadata, 'sformCode', 'sform_code')
    image.set_qform(
        affine,
        code=None if qform_code is None else _coerce_metadata_int(qform_code, 'qformCode'),
    )
    image.set_sform(
        affine,
        code=None if sform_code is None else _coerce_metadata_int(sform_code, 'sformCode'),
    )

    vox_offset = _get_metadata_value(metadata, 'voxOffset', 'vox_offset')
    if vox_offset is not None:
        image.header['vox_offset'] = _coerce_metadata_float(vox_offset, 'voxOffset')

    description = _get_metadata_value(metadata, 'description', 'descrip')
    if description is not None:
        image.header['descrip'] = str(description).encode('utf-8')[:80]

    return image
