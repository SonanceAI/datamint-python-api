""""
Test nnU-Net prediction pipeline using NNUNetInferenceModel. This includes loading the model context, running inference on a mock resource, and verifying that the output segmentation has the expected class map. 
We also check that temporary directories are cleaned up after prediction.
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import MagicMock, patch
from datamint.lightning.trainers.specialized.nnunet.inference_model import NNUNetInferenceModel

# Fake nnunetv2 submodules so patch() can resolve them
def _mock_nnunetv2():
    root = MagicMock()
    root.__version__ = '2.4.0'
    inference_mod = MagicMock()
    pred_mod = MagicMock()
    root.inference = inference_mod
    inference_mod.predict_from_raw_data = pred_mod
    sys.modules.setdefault('nnunetv2', root)
    sys.modules.setdefault('nnunetv2.inference', inference_mod)
    sys.modules.setdefault('nnunetv2.inference.predict_from_raw_data', pred_mod)

_mock_nnunetv2()

def _make_bundle(tmp_path, fold=0):
    bundle = tmp_path / 'nnunet_bundle'
    fold_dir = bundle / f'fold_{fold}'
    fold_dir.mkdir(parents=True)
    (bundle / 'nnUNetPlans.json').write_text('{}')
    (bundle / 'dataset_fingerprint.json').write_text('{}')
    (fold_dir / 'checkpoint_best.pth').write_bytes(b'fake')
    return bundle


def _make_context(bundle_path):
    ctx = MagicMock()
    ctx.artifacts = {'nnunet_bundle': str(bundle_path)}
    ctx.model_config = {}
    return ctx


def _make_resource(shape=(64, 64, 32)):
    arr = np.zeros(shape, dtype=np.int16)
    resource = MagicMock()
    resource.id = 'res-001'
    resource.fetch_file_data.return_value = nib.Nifti1Image(arr, np.eye(4))
    return resource


def test_load_context_initializes_predictor(tmp_path):
    bundle = _make_bundle(tmp_path)
    model = NNUNetInferenceModel(class_map={1: 'liver'}, configuration='3d_fullres', folds=(0,))
    with patch('nnunetv2.inference.predict_from_raw_data.nnUNetPredictor') as MockPred:
        model.load_context(_make_context(bundle))
    MockPred.return_value.initialize_from_trained_model_folder.assert_called_once_with(
        model_training_output_dir=str(bundle),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    assert model._predictor is not None


def test_predict_volume_returns_volume_segmentation(tmp_path):
    bundle = _make_bundle(tmp_path)
    model = NNUNetInferenceModel(class_map={1: 'liver'}, configuration='3d_fullres', folds=(0,))

    def fake_predict(src, dst, **kw):
        arr = np.zeros((64, 64, 32), dtype=np.int32)
        arr[10:20, 10:20, 5:15] = 1
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(Path(dst) / 'case_001.nii.gz'))

    with patch('nnunetv2.inference.predict_from_raw_data.nnUNetPredictor') as MockPred:
        MockPred.return_value.predict_from_files.side_effect = fake_predict
        model.load_context(_make_context(bundle))
        results = model.predict_volume([_make_resource()])

    assert len(results) == 1 and results[0][0].class_map == {1: 'liver'}


def test_predict_volume_cleans_up_temp_dir(tmp_path):
    bundle = _make_bundle(tmp_path)
    model = NNUNetInferenceModel(class_map={1: 'liver'}, configuration='3d_fullres', folds=(0,))
    captured = []

    def fake_predict(src, dst, **kw):
        captured.append(dst)
        arr = np.zeros((64, 64, 32), dtype=np.int32)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(Path(dst) / 'case_001.nii.gz'))

    with patch('nnunetv2.inference.predict_from_raw_data.nnUNetPredictor') as MockPred:
        MockPred.return_value.predict_from_files.side_effect = fake_predict
        model.load_context(_make_context(bundle))
        model.predict_volume([_make_resource()])

    assert captured and not Path(captured[0]).exists()
