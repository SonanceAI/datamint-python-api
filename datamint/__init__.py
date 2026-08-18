"""
Datamint API package alias.
"""

import importlib.metadata
from typing import TYPE_CHECKING

from .utils.logging_utils import setup_file_logging_if_enabled

setup_file_logging_if_enabled()
if TYPE_CHECKING:
    from .api.client import Api as Api

    from .dataset.base import DatamintBaseDataset as DatamintBaseDataset
    from .dataset.base import DatamintDatasetException as DatamintDatasetException
    from .dataset.factory import build_dataset as build_dataset
    from .dataset.image_dataset import ImageDataset as ImageDataset
    from .dataset.multiframe_dataset import MultiFrameDataset as MultiFrameDataset
    from .dataset.sliced_dataset import SlicedVolumeDataset as SlicedVolumeDataset
    from .dataset.sliced_video_dataset import SlicedVideoDataset as SlicedVideoDataset
    from .dataset.split_result import SplitResult as SplitResult
    from .dataset.video_dataset import VideoDataset as VideoDataset
    from .dataset.volume_dataset import VolumeDataset as VolumeDataset

    from .default_project import select_project as select_project
    from .importers.coco import COCOImporter as COCOImporter
    from .importers.pascal_voc import PascalVOCImporter as PascalVOCImporter
    from .importers.yolo import YOLOImporter as YOLOImporter
    from .mlflow.flavors.validation import (
        ModelValidationError as ModelValidationError,
        ValidationIssue as ValidationIssue,
        ValidationReport as ValidationReport,
        validate_model as validate_model,
    )

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=['dataset', "examples", "importers"],
        submod_attrs={
            "api.client": ["Api"],
            # New modular dataset classes
            "dataset.base": ["DatamintBaseDataset", "DatamintDatasetException"],
            "dataset.factory": ["build_dataset"],
            "dataset.image_dataset": ["ImageDataset"],
            "dataset.multiframe_dataset": ["MultiFrameDataset"],
            "dataset.sliced_dataset": ["SlicedVolumeDataset"],
            "dataset.sliced_video_dataset": ["SlicedVideoDataset"],
            "dataset.split_result": ["SplitResult"],
            "dataset.video_dataset": ["VideoDataset"],
            "dataset.volume_dataset": ["VolumeDataset"],
            "mlflow.flavors.validation": ["validate_model", "ValidationReport",
                                          "ValidationIssue", "ModelValidationError"],
            "default_project": ["select_project"],
            "importers.coco": ["COCOImporter"],
            "importers.pascal_voc": ["PascalVOCImporter"],
            "importers.yolo": ["YOLOImporter"],
        },
    )

__name__ = "datamint"
__version__ = importlib.metadata.version(__name__)