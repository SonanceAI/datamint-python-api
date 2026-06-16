"""Tests for DetectionDataset and detection_collate_fn."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from datamint.dataset.detection_dataset import DetectionDataset, detection_collate_fn


# -- Helpers

def _make_ann(x1, y1, x2, y2, cls, frame_index=None):
    return SimpleNamespace(
        annotation_type='square',
        identifier=cls,
        frame_index=frame_index,
        geometry=SimpleNamespace(point1=(x1, y1, 0), point2=(x2, y2, 0)),
    )


def _make_resource(rid, boxes=None, frame_index=None):
    """boxes: list of (x1, y1, x2, y2, class_name) tuples."""
    annotations = [
        _make_ann(x1, y1, x2, y2, cls, frame_index=frame_index)
        for (x1, y1, x2, y2, cls) in (boxes or [])
    ]
    return SimpleNamespace(
        id=rid,
        fetch_annotations=lambda annotation_type=None: annotations,
    )


def _make_dataset(resources):
    """Build a DetectionDataset without hitting the network."""
    ds = object.__new__(DetectionDataset)
    resource_annotations = [list(r.fetch_annotations()) for r in resources]
    ds.__dict__.update({
        'resources': resources,
        'resource_annotations': resource_annotations,
        'project': None,
        'split_name': None,
        'split_source': None,
        'split_as_of_timestamp': None,
        '_is_prepared': True,
        '_DatamintBaseDataset__api': None,
        '_server_url': None,
        '_api_key': None,
        '_auto_update': False,
        '_logged_uint16_conversion': False,
        'return_segmentations': False,
        'alb_transform': None,
    })
    ds._class_map = ds._build_class_map()
    return ds


# -- __getitem tests

def test_getitem_returns_dict_with_expected_keys():
    r = _make_resource('r1', boxes=[(10, 20, 50, 60, 'nodule')])
    ds = _make_dataset([r])
    ds._load_image = lambda idx: np.zeros((128, 128, 3), dtype=np.uint8)
    item = ds[0]
    assert set(item.keys()) >= {'image', 'boxes', 'labels', 'resource_id', 'identifiers'}


def test_boxes_shape_and_dtype():
    r = _make_resource('r1', boxes=[(10, 20, 50, 60, 'nodule'), (5, 5, 30, 30, 'nodule')])
    ds = _make_dataset([r])
    ds._load_image = lambda idx: np.zeros((128, 128, 3), dtype=np.uint8)
    item = ds[0]
    assert item['boxes'].shape == (2, 4)
    assert item['boxes'].dtype == torch.float32


def test_resource_with_no_boxes_returns_empty_tensors():
    r = _make_resource('r1', boxes=[])
    ds = _make_dataset([r])
    ds._load_image = lambda idx: np.zeros((128, 128, 3), dtype=np.uint8)
    item = ds[0]
    assert item['boxes'].shape == (0, 4)
    assert item['labels'].shape == (0,)


def test_image_shape_is_chw():
    r = _make_resource('r1', boxes=[(0, 0, 10, 10, 'x')])
    ds = _make_dataset([r])
    ds._load_image = lambda idx: np.zeros((64, 128, 3), dtype=np.uint8)
    item = ds[0]
    # (C, H, W) = (3, 64, 128)
    assert item['image'].shape == (3, 64, 128)


def test_resource_id_matches():
    r = _make_resource('abc-123', boxes=[])
    ds = _make_dataset([r])
    ds._load_image = lambda idx: np.zeros((32, 32, 3), dtype=np.uint8)
    item = ds[0]
    assert item['resource_id'] == 'abc-123'


# -- build_class_map tests

def test_class_map_is_stable_across_instances():
    r1 = _make_resource('r1', boxes=[(0, 0, 10, 10, 'tumor')])
    r2 = _make_resource('r2', boxes=[(0, 0, 10, 10, 'cyst')])
    ds = _make_dataset([r1, r2])
    # alphabetical order: cyst=0, tumor=1
    assert ds._class_map == {'cyst': 0, 'tumor': 1}


def test_class_map_empty_when_no_boxes():
    r = _make_resource('r1', boxes=[])
    ds = _make_dataset([r])
    assert ds._class_map == {}


# -- _fetch_boxes tests

def test_fetch_boxes_returns_correct_coords():
    ann = _make_ann(10, 20, 50, 60, 'nodule')
    r = SimpleNamespace(id='r1', fetch_annotations=lambda annotation_type=None: [ann])
    ds = _make_dataset([r])
    boxes = ds._fetch_boxes(r)
    assert len(boxes) == 1
    assert boxes[0] == pytest.approx((10.0, 20.0, 50.0, 60.0))


def test_frame_index_filters_boxes():
    ann_f0 = _make_ann(10, 10, 40, 40, 'nodule', frame_index=0)
    ann_f1 = _make_ann(50, 50, 80, 80, 'nodule', frame_index=1)
    r = SimpleNamespace(id='r1', fetch_annotations=lambda annotation_type=None: [ann_f0, ann_f1])
    ds = _make_dataset([r])
    boxes = ds._fetch_boxes(r, frame_index=0)
    assert len(boxes) == 1
    assert boxes[0][0] == 10  # x1 of ann_f0


def test_fetch_boxes_no_frame_filter_returns_all():
    ann_f0 = _make_ann(10, 10, 40, 40, 'nodule', frame_index=0)
    ann_f1 = _make_ann(50, 50, 80, 80, 'nodule', frame_index=1)
    r = SimpleNamespace(id='r1', fetch_annotations=lambda annotation_type=None: [ann_f0, ann_f1])
    ds = _make_dataset([r])
    boxes = ds._fetch_boxes(r, frame_index=None)
    assert len(boxes) == 2


# -- detection_collate_fn tests

def test_collate_fn_stacks_images():
    batch = [
        {'image': torch.zeros(3, 64, 64), 'boxes': torch.zeros(2, 4),
         'labels': torch.zeros(2, dtype=torch.int64),
         'resource_id': 'r1', 'identifiers': ['a', 'b']},
        {'image': torch.zeros(3, 64, 64), 'boxes': torch.zeros(0, 4),
         'labels': torch.zeros(0, dtype=torch.int64),
         'resource_id': 'r2', 'identifiers': []},
    ]
    collated = detection_collate_fn(batch)
    assert collated['image'].shape == (2, 3, 64, 64)


def test_collate_fn_keeps_boxes_as_list():
    batch = [
        {'image': torch.zeros(3, 64, 64), 'boxes': torch.zeros(3, 4),
         'labels': torch.zeros(3, dtype=torch.int64),
         'resource_id': 'r1', 'identifiers': ['a', 'b', 'c']},
        {'image': torch.zeros(3, 64, 64), 'boxes': torch.zeros(1, 4),
         'labels': torch.zeros(1, dtype=torch.int64),
         'resource_id': 'r2', 'identifiers': ['a']},
    ]
    collated = detection_collate_fn(batch)
    assert isinstance(collated['boxes'], list)
    assert collated['boxes'][0].shape == (3, 4)
    assert collated['boxes'][1].shape == (1, 4)


def test_collate_fn_resource_ids_are_list():
    batch = [
        {'image': torch.zeros(3, 32, 32), 'boxes': torch.zeros(0, 4),
         'labels': torch.zeros(0, dtype=torch.int64),
         'resource_id': 'r1', 'identifiers': []},
    ]
    collated = detection_collate_fn(batch)
    assert collated['resource_id'] == ['r1']
