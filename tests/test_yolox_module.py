"""Tests for YOLOXModule."""
import sys
import torch
import pytest
from unittest.mock import MagicMock, patch

# Mock yolox before any import touches it. sys.modules['yolox.models'] must be
# the same object as getattr(yolox_mock, 'models') so that both `import yolox.models`
# (production code) and patch('yolox.models.yolox_s') (tests) resolve to the same object.
_yolox_mock = MagicMock()
sys.modules.setdefault('yolox', _yolox_mock)
sys.modules.setdefault('yolox.models', _yolox_mock.models)
sys.modules.setdefault('yolox.utils', _yolox_mock.utils)

from datamint.lightning.trainers.lightning_modules.detection_modules.yolox_module import YOLOXModule
from datamint.entities.annotations import BoxAnnotation

@pytest.fixture()
def module():
    with patch('yolox.models.yolox_s') as MockCtor:
        MockCtor.return_value = MagicMock()
        m = YOLOXModule(num_classes=2, model_size='s')
    m.model = MagicMock()
    m.map_metric = None 
    return m

# -- testing __init__ and model_size handling --
def test_invalid_model_size_raises():
    with pytest.raises(ValueError, match='model_size'):
        with patch('yolox.models.yolox_bad', create=True):
            YOLOXModule(num_classes=1, model_size='bad')


def test_model_size_routes_to_correct_variant():
    for size in ('nano', 'tiny', 's', 'm', 'l', 'x'):
        target = f'yolox.models.yolox_{size}'
        with patch(target) as MockVariant:
            MockVariant.return_value = MagicMock()
            m = YOLOXModule(num_classes=1, model_size=size)
        MockVariant.assert_called_once_with(num_classes=1)


# -- testing _build_targets --

def test_build_targets_shape():
    boxes = [torch.tensor([[10., 20., 50., 60.]]), torch.zeros(0, 4)]
    labels = [torch.tensor([1]), torch.zeros(0, dtype=torch.int64)]
    t = YOLOXModule._build_targets(boxes, labels, torch.device('cpu'))
    assert t.shape == (2, 1, 5)


def test_build_targets_center_form():
    # x1=0, y1=0, x2=10, y2=20 → cx=5, cy=10, w=10, h=20
    boxes = [torch.tensor([[0., 0., 10., 20.]])]
    labels = [torch.tensor([0])]
    t = YOLOXModule._build_targets(boxes, labels, torch.device('cpu'))
    assert t[0, 0, 1].item() == pytest.approx(5.0)   # cx
    assert t[0, 0, 2].item() == pytest.approx(10.0)  # cy
    assert t[0, 0, 3].item() == pytest.approx(10.0)  # w
    assert t[0, 0, 4].item() == pytest.approx(20.0)  # h


def test_build_targets_class_index():
    boxes = [torch.tensor([[0., 0., 10., 10.]])]
    labels = [torch.tensor([3])]
    t = YOLOXModule._build_targets(boxes, labels, torch.device('cpu'))
    assert t[0, 0, 0].item() == pytest.approx(3.0)


def test_build_targets_empty_batch_no_crash():
    boxes = [torch.zeros(0, 4), torch.zeros(0, 4)]
    labels = [torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)]
    t = YOLOXModule._build_targets(boxes, labels, torch.device('cpu'))

    assert t.shape == (2, 1, 5)
    assert t.sum().item() == pytest.approx(0.0)


# -- testing training_step --

def test_training_step_logs_train_loss(module):
    module.model.return_value = {
        'total_loss': torch.tensor(1.5),
        'iou_loss': torch.tensor(0.5),
        'cls_loss': torch.tensor(0.3),
        'conf_loss': torch.tensor(0.4),  
    }
    batch = {
        'image': torch.zeros(2, 3, 416, 416),
        'boxes': [torch.zeros(2, 4), torch.zeros(1, 4)],
        'box_labels': [torch.zeros(2, dtype=torch.int64), torch.zeros(1, dtype=torch.int64)],
    }
    with patch.object(module, 'log') as mock_log:
        loss = module.training_step(batch, 0)

    logged_keys = [c.args[0] for c in mock_log.call_args_list]
    assert 'train/loss' in logged_keys
    assert 'train/iou_loss' in logged_keys
    assert 'train/cls_loss' in logged_keys
    assert 'train/obj_loss' in logged_keys
    assert loss.item() == pytest.approx(1.5)


def test_training_step_returns_total_loss(module):
    module.model.return_value = {
        'total_loss': torch.tensor(2.0),
        'iou_loss': torch.tensor(0.5),
        'cls_loss': torch.tensor(0.5),
        'conf_loss': torch.tensor(0.5),
    }
    batch = {
        'image': torch.zeros(1, 3, 416, 416),
        'boxes': [torch.zeros(0, 4)],
        'box_labels': [torch.zeros(0, dtype=torch.int64)],
    }
    with patch.object(module, 'log'):
        loss = module.training_step(batch, 0)
    assert loss.item() == pytest.approx(2.0)


# -- testing predict_image --

def _make_fake_resource(img_np=None):
    """Return a mock resource whose fetch_file_data yields a small numpy image."""
    import numpy as np
    from unittest.mock import MagicMock
    if img_np is None:
        img_np = np.zeros((64, 64, 3), dtype=np.uint8)
    res = MagicMock()
    res.fetch_file_data.return_value = img_np
    return res


def test_predict_image_returns_box_annotations(module):
    import numpy as np
    fake_det = torch.tensor([[10.0, 20.0, 50.0, 60.0, 0.9, 0.85, 0.0]])
    with patch('yolox.utils.postprocess', return_value=[fake_det]):
        results = module.predict_image([_make_fake_resource()])
    assert len(results) == 1           # one list per resource
    assert len(results[0]) == 1        # one detection
    assert isinstance(results[0][0], BoxAnnotation)


def test_predict_image_empty_returns_empty_list(module):
    with patch('yolox.utils.postprocess', return_value=[None]):
        results = module.predict_image([_make_fake_resource()])
    assert results == [[]]


def test_predict_image_box_coordinates(module):
    fake_det = torch.tensor([[5.0, 10.0, 55.0, 70.0, 0.9, 0.9, 1.0]])
    with patch('yolox.utils.postprocess', return_value=[fake_det]):
        results = module.predict_image([_make_fake_resource()])
    ann = results[0][0]
    x1, y1, _ = ann.geometry.point1
    x2, y2, _ = ann.geometry.point2
    assert x1 == pytest.approx(5.0)
    assert y1 == pytest.approx(10.0)
    assert x2 == pytest.approx(55.0)
    assert y2 == pytest.approx(70.0)


def test_predict_image_class_identifier(module):
    fake_det = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9, 0.9, 2.0]])
    with patch('yolox.utils.postprocess', return_value=[fake_det]):
        results = module.predict_image([_make_fake_resource()])
    assert results[0][0].identifier == '2'


def test_predict_image_multiple_resources(module):
    import numpy as np
    fake_det = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9, 0.9, 0.0]])
    resources = [_make_fake_resource(), _make_fake_resource()]
    with patch('yolox.utils.postprocess', side_effect=[[fake_det], [None]]):
        results = module.predict_image(resources)
    assert len(results) == 2
    assert len(results[0]) == 1
    assert len(results[1]) == 0
