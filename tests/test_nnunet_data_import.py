"""
Test the NNUNetToDatamintImporter class for importing nnUNet predictions into Datamint.
"""
import pytest
pytest.importorskip("nnunetv2", minversion="2.4")
import json
import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from datamint.lightning.trainers.specialized.nnunet.data_import import NNUNetToDatamintImporter


def _write_pred(path: Path, shape=(64, 64, 32)):
    arr = np.zeros(shape, dtype=np.int32)
    arr[10:20, 10:20, 5:15] = 1
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))


@pytest.fixture()
def setup(tmp_path):
    pred_dir = tmp_path / 'predictions'
    pred_dir.mkdir()
    _write_pred(pred_dir / 'case_001.nii.gz')

    dataset_dir = tmp_path / 'Dataset001_CTLiver'
    dataset_dir.mkdir()
    (dataset_dir / 'datamint_case_map.json').write_text(
        json.dumps({'case_001': 'res-uuid-001'})
    )

    api = MagicMock()
    return api, dataset_dir, pred_dir


def test_import_creates_volume_segmentation(setup):
    api, dataset_dir, pred_dir = setup
    segs = NNUNetToDatamintImporter(api, dataset_dir).import_predictions(
        pred_dir, class_map={1: 'liver'}
    )
    assert len(segs) == 1
    assert segs[0].class_map == {1: 'liver'}


def test_import_uploads_to_correct_resource(setup):
    api, dataset_dir, pred_dir = setup
    NNUNetToDatamintImporter(api, dataset_dir).import_predictions(
        pred_dir, class_map={1: 'liver'}
    )
    assert api.annotations.upload_volume_segmentation.call_args.kwargs['resource'] == 'res-uuid-001'


def test_import_raises_on_missing_case(setup):
    api, dataset_dir, pred_dir = setup
    _write_pred(pred_dir / 'case_999.nii.gz')
    with pytest.raises(KeyError, match='case_999'):
        NNUNetToDatamintImporter(api, dataset_dir).import_predictions(
            pred_dir, class_map={1: 'liver'}
        )
