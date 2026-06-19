from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datamint.entities.resource import Resource
    from .base import DatamintBaseDataset

_LOGGER = logging.getLogger(__name__)


def _classify_resource(resource: 'Resource') -> str:
    if resource.is_video():
        return 'video'
    if resource.is_volume():
        try:
            if resource.get_depth() == 1:
                return 'image'
        except Exception:
            pass  # frame_count not in list-response metadata; assume volume
        return 'volume'
    if resource.is_image():
        return 'image'
    return getattr(resource, 'kind', 'unknown')


def build_dataset(project_name: str, **kwargs: Any) -> 'DatamintBaseDataset':
    """Auto-detect and return the appropriate dataset class for a project.

    Fetches a small sample of resources from the project to determine the
    data type, then instantiates and returns the matching dataset class:

    - 2D images (JPEG, PNG) → :class:`~datamint.dataset.ImageDataset`
    - Volumes (NIfTI, DICOM, volumetric) → :class:`~datamint.dataset.VolumeDataset`
    - Videos → :class:`~datamint.dataset.VideoDataset`

    Args:
        project_name: Name of the Datamint project.
        **kwargs: Forwarded to the dataset constructor (transforms, filters, etc.).

    Returns:
        An instantiated dataset of the detected type.

    Raises:
        ValueError: If the project is empty, contains unknown resource types,
            or contains a mix of data types.

    Example::

        from datamint.dataset import build_dataset

        ds = build_dataset('MyProject', include_unannotated=False)
    """
    from datamint import Api
    from .image_dataset import ImageDataset
    from .volume_dataset import VolumeDataset
    from .video_dataset import VideoDataset

    _KIND_TO_CLS = {
        'image': ImageDataset,
        'volume': VolumeDataset,
        'nifti': VolumeDataset,
        'dicom': VolumeDataset,
        'video': VideoDataset,
    }

    api = Api()
    sample = api.resources.get_list(project_name=project_name, limit=5)

    if not sample:
        raise ValueError(f"Project '{project_name}' has no resources.")

    kinds = {_classify_resource(r) for r in sample}
    unknown = kinds - set(_KIND_TO_CLS)
    if unknown:
        raise ValueError(
            f"Project '{project_name}' contains unsupported resource types: {sorted(unknown)}. "
            "Instantiate the dataset class directly."
        )

    if kinds == {'image', 'volume'}:
        _LOGGER.warning(
            f"Project '{project_name}' contains a mix of 2D and 3D DICOM resources. "
            "Defaulting to VolumeDataset. Use ImageDataset or VolumeDataset directly "
            "if you need a specific type."
        )
        kinds = {'volume'}

    if len(kinds) > 1:
        raise ValueError(
            f"Project '{project_name}' contains mixed data types: {sorted(kinds)}. "
            "Instantiate the dataset class directly."
        )

    kind = next(iter(kinds))
    dataset_cls = _KIND_TO_CLS[kind]
    _LOGGER.info(f"Detected resource type '{kind}'; using {dataset_cls.__name__}.")
    return dataset_cls(project=project_name, **kwargs)
