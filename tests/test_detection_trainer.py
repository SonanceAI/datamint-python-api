"""Tests for DetectionTrainer abstract base."""
import pytest
from unittest.mock import MagicMock, patch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datamint.lightning.trainers.detection_trainer import DetectionTrainer
from datamint.dataset.image_dataset import ImageDataset, detection_collate_fn
from datamint.lightning.datamodule import DatamintDataModule


class _ConcreteDetectionTrainer(DetectionTrainer):
    """Minimal concrete subclass for testing the abstract base."""

    def _build_model(self, loss_fn, metrics):
        return MagicMock()

    def _train_transform(self):
        return A.Compose(
            [ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        )

    def _eval_transform(self):
        return self._train_transform()


def _bare_trainer():
    """Return an uninitialised trainer (skips BaseTrainer.__init__)."""
    return _ConcreteDetectionTrainer.__new__(_ConcreteDetectionTrainer)


# -- _build_dataset tests

def test_build_dataset_returns_image_dataset():
    trainer = _bare_trainer()
    with patch.object(ImageDataset, '__init__', return_value=None):
        ds = trainer._build_dataset(project='thyroid')
    assert isinstance(ds, ImageDataset)


def test_build_dataset_passes_project_kwarg():
    trainer = _bare_trainer()
    with patch.object(ImageDataset, '__init__', return_value=None) as mock_init:
        trainer._build_dataset(project='thyroid')
    assert mock_init.call_args.kwargs.get('project') == 'thyroid'


def test_build_dataset_sets_return_boxes():
    trainer = _bare_trainer()
    with patch.object(ImageDataset, '__init__', return_value=None) as mock_init:
        trainer._build_dataset(project='thyroid')
    assert mock_init.call_args.kwargs.get('return_boxes') is True


# -- _build_datamodule tests

def test_datamodule_receives_detection_collate_fn():
    trainer = _bare_trainer()
    trainer.batch_size = 8
    trainer.num_workers = 0
    trainer.split_as_of_timestamp = None

    with patch.object(DatamintDataModule, '__init__', return_value=None) as mock_init:
        trainer._build_datamodule(MagicMock(), MagicMock(), MagicMock())

    assert mock_init.call_args.kwargs.get('collate_fn') is detection_collate_fn


# -- metrics tests

def test_loss_returns_none():
    trainer = _bare_trainer()
    assert trainer._loss() is None


def test_monitor_metric_is_map():
    trainer = _bare_trainer()
    metric, mode = trainer._monitor_metric()
    assert metric == 'val/map'
    assert mode == 'max'
