"""
Testing the DatamintToNNUNetExporter class to ensure it correctly writes dataset.json, preserves voxel spacing, merges segmentations, and creates the expected directory structure for nnUNet.
"""
import json
import numpy as np
import nibabel as nib
import pytest
from unittest.mock import MagicMock
from datamint.lightning.trainers.specialized.nnunet.data_export import DatamintToNNUNetExporter


def _make_nifti(shape=(64, 64, 32), zooms=(1.5, 1.5, 2.0)) -> nib.Nifti1Image:
    data = np.zeros(shape, dtype=np.int16)
    affine = np.diag([*zooms, 1.0])
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms(zooms)
    return img


def _make_resource(resource_id: str, nifti: nib.Nifti1Image) -> MagicMock:
    resource = MagicMock()
    resource.id = resource_id
    resource.fetch_file_data.return_value = nifti
    return resource


def _make_seg(label_value: int, shape=(64, 64, 32)) -> MagicMock:
    arr = np.zeros(shape, dtype=np.int32)
    arr[10:20, 10:20, 5:15] = 1  
    seg = MagicMock()
    seg.fetch_file_data.return_value = arr
    seg.identifier = 'liver' if label_value == 1 else 'tumor'
    seg.class_map = {label_value: seg.identifier}
    return seg


def test_dataset_json_is_written(tmp_path):
    exp = DatamintToNNUNetExporter(tmp_path, dataset_id=1, dataset_name='CTLiver')
    exp._write_dataset_json(
        labels={'background': 0, 'liver': 1},
        channel_names={'0': 'CT'},
        num_training=10,
    )
    content = json.loads((tmp_path / 'Dataset001_CTLiver' / 'dataset.json').read_text())
    assert content['labels'] == {'background': 0, 'liver': 1}
    assert content['channel_names'] == {'0': 'CT'}
    assert content['numTraining'] == 10
    assert content['file_ending'] == '.nii.gz'


def test_export_image_preserves_voxel_spacing(tmp_path):
    zooms = (1.5, 1.5, 2.0)
    resource = _make_resource('res-001', _make_nifti(zooms=zooms))
    exp = DatamintToNNUNetExporter(tmp_path, dataset_id=1, dataset_name='CTLiver')
    exp._export_image(resource, case_id='case_001', split='train')

    out = tmp_path / 'Dataset001_CTLiver' / 'imagesTr' / 'case_001_0000.nii.gz'
    assert out.exists()
    assert nib.load(str(out)).header.get_zooms() == pytest.approx(zooms, abs=1e-4)


def test_merge_segmentations_highest_class_wins(tmp_path):
    shape = (32, 32, 16)
    liver = np.zeros(shape, dtype=np.int32); liver[5:15, 5:15, 2:8] = 1
    tumor = np.zeros(shape, dtype=np.int32); tumor[10:20, 10:20, 4:10] = 2

    liver_seg = MagicMock(); liver_seg.fetch_file_data.return_value = liver
    tumor_seg = MagicMock(); tumor_seg.fetch_file_data.return_value = tumor

    exp = DatamintToNNUNetExporter(tmp_path, dataset_id=1, dataset_name='CTLiver')
    merged = exp._merge_segmentations([liver_seg, tumor_seg])

    assert merged[12, 12, 5] == 2   # overlap → tumor wins
    assert merged[6, 6, 3] == 1     # no overlap → liver preserved


def test_merge_segmentations_warns_on_overlap(tmp_path):
    shape = (32, 32, 16)
    seg1 = MagicMock(); seg1.fetch_file_data.return_value = np.ones(shape, dtype=np.int32)
    seg2 = MagicMock(); seg2.fetch_file_data.return_value = np.ones(shape, dtype=np.int32) * 2

    exp = DatamintToNNUNetExporter(tmp_path, dataset_id=1, dataset_name='CTLiver')
    with pytest.warns(UserWarning, match='overlap'):
        exp._merge_segmentations([seg1, seg2])


def test_case_map_is_written(tmp_path):
    exp = DatamintToNNUNetExporter(tmp_path, dataset_id=1, dataset_name='CTLiver')
    mapping = {'case_001': 'res-uuid-001', 'case_002': 'res-uuid-002'}
    exp._write_case_map(mapping)
    loaded = json.loads(
        (tmp_path / 'Dataset001_CTLiver' / 'datamint_case_map.json').read_text()
    )
    assert loaded == mapping


def test_export_creates_full_directory_structure(tmp_path):
    nifti = _make_nifti()
    resources = [_make_resource(f'res-00{i}', nifti) for i in range(3)]
    for r in resources:
        r.fetch_annotations.return_value = [_make_seg(1)]

    split = {'train': [resources[0], resources[1]], 'test': [resources[2]]}
    exp = DatamintToNNUNetExporter(tmp_path, dataset_id=1, dataset_name='CTLiver')
    dataset_path = exp.export(split, channel_names={'0': 'CT'})

    assert (dataset_path / 'dataset.json').exists()
    assert (dataset_path / 'imagesTr' / 'case_001_0000.nii.gz').exists()
    assert (dataset_path / 'labelsTr' / 'case_001.nii.gz').exists()
    assert (dataset_path / 'imagesTs' / 'case_003_0000.nii.gz').exists()
    assert not (dataset_path / 'labelsTs').exists()

    label_arr = nib.load(str(dataset_path / 'labelsTr' / 'case_001.nii.gz')).get_fdata()
    assert label_arr[15, 15, 7] == 1   # liver voxel scaled to class index 1
    assert label_arr[0, 0, 0] == 0     # background preserved
