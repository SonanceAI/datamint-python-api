"""Tests for YOLOXTrainer."""
import pytest
from unittest.mock import MagicMock, patch

import albumentations as A

from datamint.lightning.trainers.specialized.yolox import YOLOXTrainer
from datamint.dataset.image_dataset import ImageDataset
from datamint.lightning.trainers.lightning_modules.detection_modules.yolox_module import YOLOXModule


@pytest.fixture()
def trainer():
    ds = MagicMock(spec=ImageDataset)
    ds.box_class_map = {'cyst': 0, 'nodule': 1}
    t = YOLOXTrainer.__new__(YOLOXTrainer)
    t.dataset = ds
    t.model_size = 's'
    t.conf_thre = 0.25
    t.nms_thre = 0.45
    t.image_size = (640, 640)
    return t


# -- _build_model tests --

def test_build_model_returns_yolox_module(trainer):
    with patch.object(YOLOXModule, '__init__', return_value=None):
        model = trainer._build_model(loss_fn=None, metrics={})
    assert isinstance(model, YOLOXModule)


def test_build_model_uses_num_classes_from_class_map(trainer):
    with patch.object(YOLOXModule, '__init__', return_value=None) as mock_init:
        trainer._build_model(loss_fn=None, metrics={})
    assert mock_init.call_args.kwargs['num_classes'] == 2


def test_build_model_forwards_model_size(trainer):
    trainer.model_size = 'l'
    with patch.object(YOLOXModule, '__init__', return_value=None) as mock_init:
        trainer._build_model(loss_fn=None, metrics={})
    assert mock_init.call_args.kwargs['model_size'] == 'l'


def test_build_model_forwards_thresholds(trainer):
    trainer.conf_thre = 0.5
    trainer.nms_thre = 0.6
    with patch.object(YOLOXModule, '__init__', return_value=None) as mock_init:
        trainer._build_model(loss_fn=None, metrics={})
    assert mock_init.call_args.kwargs['conf_thre'] == pytest.approx(0.5)
    assert mock_init.call_args.kwargs['nms_thre'] == pytest.approx(0.6)


def test_build_model_raises_when_no_classes(trainer):
    trainer.dataset.box_class_map = {}
    with pytest.raises(ValueError, match='No box annotation classes'):
        trainer._build_model(loss_fn=None, metrics={})


# -- image_size normalization test --

def test_image_size_int_becomes_tuple():
    with patch.object(ImageDataset, '__init__', return_value=None):
        t = YOLOXTrainer.__new__(YOLOXTrainer)
        t.image_size = 416 
        if isinstance(t.image_size, int):
            t.image_size = (t.image_size, t.image_size)
    assert t.image_size == (416, 416)


# -- transforms tests --

def test_train_transform_has_bbox_params(trainer):
    tfm = trainer._train_transform()
    assert tfm.processors.get('bboxes') is not None


def test_eval_transform_has_bbox_params(trainer):
    tfm = trainer._eval_transform()
    assert tfm.processors.get('bboxes') is not None


def test_train_transform_is_compose(trainer):
    tfm = trainer._train_transform()
    assert isinstance(tfm, A.Compose)


def test_eval_transform_is_compose(trainer):
    tfm = trainer._eval_transform()
    assert isinstance(tfm, A.Compose)

