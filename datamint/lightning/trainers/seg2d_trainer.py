"""2-D semantic segmentation trainer."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TYPE_CHECKING, cast

import lightning as L
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datamint.dataset import ImageDataset, SlicedVolumeDataset
from datamint.utils.nifti_utils import metadata_to_nifti_obj

from .segmentation_trainer import SegmentationTrainer

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from nibabel.spatialimages import SpatialImage
    from pydicom import Dataset as DicomDataset
    from medimgkit import ViewPlane

    from datamint.entities import Project, Resource

import logging

_LOGGER = logging.getLogger(__name__)

SliceAxisName = Literal['axial', 'coronal', 'sagittal']
_VALID_SLICE_AXES: tuple[SliceAxisName, ...] = ('axial', 'coronal', 'sagittal')


class SemanticSegmentation2DTrainer(SegmentationTrainer):
    """Trainer for 2-D semantic segmentation.

    Default model: **UNet++** (``segmentation_models_pytorch``) with a
    ``resnet34`` encoder pretrained on ImageNet.

    When pointed at a project made of 3-D volumes, the trainer automatically
    converts it to a :class:`~datamint.dataset.SlicedVolumeDataset` and trains
    on 2-D slices instead.

    Args:
        slice_axis: Slice axis override for 3-D volume projects. When omitted,
            the trainer tries to infer the most sensible anatomical plane and
            falls back to ``'axial'``.
        in_channels: Number of input image channels.  Defaults to ``3``.
        All remaining keyword arguments are forwarded to
        :class:`~datamint.lightning.trainers.base_trainer.BaseTrainer`.

    Example::

        trainer = SemanticSegmentation2DTrainer(project='BUS_Segmentation')
        results = trainer.fit()
    """

    def __init__(
        self,
        *,
        image_size: int | tuple[int, int] | None = None,
        slice_axis: 'ViewPlane | int | None' = None,
        model: L.LightningModule | type[L.LightningModule] | None = None,
        in_channels: int = 3,
        trainer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model,
                         trainer_kwargs=trainer_kwargs,
                         **kwargs)
        self.in_channels = in_channels
        self.slice_axis: 'ViewPlane | int | None' = slice_axis
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

    def _build_dataset(self, project: 'str | Project') -> ImageDataset | SlicedVolumeDataset:
        dataset = ImageDataset(
            project=project,
            return_as_semantic_segmentation=True,
            semantic_seg_merge_strategy='union',
            allow_external_annotations=True,
            include_unannotated=False,
        )
        dataset._prepare()

        resource_kinds = {self._classify_resource(resource) for resource in dataset.resources}
        unsupported_kinds = sorted(resource_kinds - {'image', 'volume'})
        if unsupported_kinds:
            raise ValueError(
                "SemanticSegmentation2DTrainer only supports projects composed entirely of 2D images "
                f"or 3D volumes. Found unsupported resource types: {unsupported_kinds}."
            )

        if len(resource_kinds) > 1:
            raise ValueError(
                "SemanticSegmentation2DTrainer requires a homogeneous project. "
                "Found a mix of 2D images and 3D volumes; build and pass a dataset manually instead."
            )

        if resource_kinds == {'volume'}:
            resolved_slice_axis = self.slice_axis if self.slice_axis is not None else self._infer_slice_axis(
                dataset.resources
            )
            _LOGGER.info(f"Project contains 3D volumes; slicing along '{resolved_slice_axis}' axis.")
            return SlicedVolumeDataset.from_dataset(dataset, slice_axis=resolved_slice_axis)

        _LOGGER.info("Project contains 2D images; using ImageDataset.")
        return dataset

    def _classify_resource(self, resource: 'Resource') -> str:
        if resource.is_video():
            return 'video'

        depth = self._get_resource_depth(resource)
        if depth == 1:
            return 'image'

        if resource.is_volume():
            return 'volume'

        if resource.is_image():
            return 'image'

        return getattr(resource, 'kind', 'unknown')

    @staticmethod
    def _get_resource_depth(resource: 'Resource') -> int | None:
        try:
            return resource.get_depth()
        except Exception:
            return None

    def _infer_slice_axis(self, resources: Sequence['Resource']) -> SliceAxisName:
        for resource in resources:
            if self._classify_resource(resource) != 'volume':
                continue

            inferred_axis = self._infer_slice_axis_from_resource(resource)
            if inferred_axis is not None:
                return inferred_axis

        return 'axial'

    def _infer_slice_axis_from_resource(self, resource: 'Resource') -> SliceAxisName | None:
        if resource.is_nifti():
            nifti_image = self._nifti_image_from_metadata(resource)
            if nifti_image is not None:
                return self._infer_slice_axis_from_nifti(nifti_image)

        try:
            volume_data = resource.fetch_file_data(auto_convert=True, use_cache=True)
        except Exception:
            return None

        try:
            if resource.is_nifti():
                return self._infer_slice_axis_from_nifti(cast('SpatialImage', volume_data))
            if resource.is_dicom():
                return self._infer_slice_axis_from_dicom(cast('DicomDataset', volume_data))
        except Exception:
            return None

        return None

    @staticmethod
    def _nifti_image_from_metadata(resource: 'Resource') -> 'SpatialImage | None':
        metadata = getattr(resource, 'metadata', None)
        if not isinstance(metadata, dict):
            return None

        try:
            return metadata_to_nifti_obj(metadata)
        except Exception:
            return None

    def _infer_slice_axis_from_nifti(self, nifti_image: 'SpatialImage') -> SliceAxisName | None:
        from medimgkit import nifti_utils

        plane_sizes = {
            plane: nifti_utils.get_dim_size(nifti_image, plane)
            for plane in _VALID_SLICE_AXES
        }
        zooms = nifti_image.header.get_zooms()
        plane_spacings: dict[str, float | None] = {
            plane: float(zooms[nifti_utils.get_plane_axis(nifti_image, plane)])
            for plane in _VALID_SLICE_AXES
            if len(zooms) > nifti_utils.get_plane_axis(nifti_image, plane)
        }
        return self._choose_slice_axis(plane_sizes, plane_spacings)

    def _infer_slice_axis_from_dicom(self, dataset: 'DicomDataset') -> SliceAxisName | None:
        from medimgkit import dicom_utils

        pixel_spacing = self._coerce_spacing_pair(getattr(dataset, 'PixelSpacing', None))
        slice_spacing = self._coerce_float(getattr(dataset, 'SpacingBetweenSlices', None))
        if slice_spacing is None:
            slice_spacing = self._coerce_float(getattr(dataset, 'SliceThickness', None))

        raw_axis_spacings: dict[int, float | None] = {
            0: slice_spacing,
            1: pixel_spacing[0] if pixel_spacing is not None else None,
            2: pixel_spacing[1] if pixel_spacing is not None else None,
        }

        plane_sizes: dict[str, int] = {}
        plane_spacings: dict[str, float | None] = {}
        for plane in _VALID_SLICE_AXES:
            axis_index = dicom_utils.get_plane_axis(dataset, plane)
            if axis_index is None:
                continue
            plane_sizes[plane] = dicom_utils.get_dim_size(dataset, axis_index)
            plane_spacings[plane] = raw_axis_spacings.get(axis_index)

        return self._choose_slice_axis(plane_sizes, plane_spacings)

    @staticmethod
    def _choose_slice_axis(
        plane_sizes: Mapping[str, int],
        plane_spacings: Mapping[str, float | None],
    ) -> SliceAxisName | None:
        best_plane: SliceAxisName | None = None
        best_key: tuple[int, float, int, int] | None = None

        for order, plane in enumerate(_VALID_SLICE_AXES):
            axis_size = plane_sizes.get(plane)
            if axis_size is None:
                continue

            spacing = plane_spacings.get(plane)
            candidate_key = (
                1 if spacing is not None else 0,
                float('-inf') if spacing is None else spacing,
                -axis_size,
                -order,
            )
            if best_key is None or candidate_key > best_key:
                best_plane = plane
                best_key = candidate_key

        return best_plane

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _coerce_spacing_pair(cls, value: Any) -> tuple[float | None, float | None] | None:
        if value is None:
            return None

        try:
            first, second = value[:2]
        except (TypeError, ValueError):
            return None

        return cls._coerce_float(first), cls._coerce_float(second)

    def _build_resize_transform(self):
        if self.image_size is None:
            return A.NoOp()
        else:
            return A.Resize(*self.image_size)

    def _train_transform(self) -> 'BaseCompose':
        return A.Compose([
            self._build_resize_transform(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),  # Imagenet stats is the default
            ToTensorV2(),
        ])

    def _eval_transform(self) -> 'BaseCompose':
        return A.Compose([
            self._build_resize_transform(),
            A.Normalize(),
            ToTensorV2(),
        ])