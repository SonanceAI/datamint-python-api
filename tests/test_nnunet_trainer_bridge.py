"""  Test the _DatamintNNUNetTrainer bridge class that extends nnUNetTrainer to log checkpoints and validation summaries to MLflow. We verify that checkpoints are logged as artifacts, that validation metrics are
logged correctly, and that the version guard prevents usage with old nnunetv2 versions."""
import json
import importlib.metadata as _meta
import pytest

try:
    _ver = tuple(int(x) for x in _meta.version('nnunetv2').split('.')[:2])
    if not ((2, 4) <= _ver < (3, 0)):
        pytest.skip("nnunetv2>=2.4,<3.0 required", allow_module_level=True)
except _meta.PackageNotFoundError:
    pytest.skip("nnunetv2 not installed", allow_module_level=True)

from unittest.mock import MagicMock, patch
import sys

# Mock nnunetv2 BEFORE importing the bridge so tests run without nnunetv2 installed.
class _FakeNNUNetTrainer:
    def save_checkpoint(self, *a, **kw): pass
    def perform_actual_validation(self, *a, **kw): pass
    def print_to_log_file(self, *a, **kw): pass

_nnunetv2_mock = MagicMock()
_nnunetv2_mock.__version__ = '2.4.0'
sys.modules.setdefault('nnunetv2', _nnunetv2_mock)
sys.modules.setdefault('nnunetv2.training', MagicMock())
sys.modules.setdefault('nnunetv2.training.nnUNetTrainer', MagicMock())

_trainer_module_mock = MagicMock()
_trainer_module_mock.nnUNetTrainer = _FakeNNUNetTrainer
sys.modules.setdefault('nnunetv2.training.nnUNetTrainer.nnUNetTrainer', _trainer_module_mock)

from datamint.lightning.trainers.specialized.nnunet._nnunet_trainer_bridge import (  
    _DatamintNNUNetTrainer,
    _MLflowLogger,
    _SKIP_METRIC_KEYS,
)


@pytest.fixture()
def bridge():
    b = _DatamintNNUNetTrainer.__new__(_DatamintNNUNetTrainer)
    b._best_checkpoint_path = None
    b.current_epoch = 5
    return b


@pytest.fixture()
def mlflow_logger():
    return _MLflowLogger()


# -- Tests for _MLflowLogger
def test_mlflow_logger_log_emits_metric(mlflow_logger):
    with patch('mlflow.log_metric') as mock_log:
        mlflow_logger.log('train_losses', 0.42, step=3)
    mock_log.assert_called_once_with('train_losses', 0.42, step=3)


def test_mlflow_logger_skips_timestamp_keys(mlflow_logger):
    with patch('mlflow.log_metric') as mock_log:
        for key in _SKIP_METRIC_KEYS:
            mlflow_logger.log(key, 12345.0, step=0)
    mock_log.assert_not_called()


def test_mlflow_logger_log_summary(mlflow_logger):
    with patch('mlflow.log_metric') as mock_log:
        mlflow_logger.log_summary('final_val/foreground_dice', 0.88)
    mock_log.assert_called_once_with('final_val/foreground_dice', 0.88)


def test_mlflow_logger_update_config_logs_params(mlflow_logger):
    with patch('mlflow.log_params') as mock_params:
        mlflow_logger.update_config({'lr': 0.01, 'epochs': 100, 'name': 'test'})
    mock_params.assert_called_once_with({'lr': 0.01, 'epochs': 100, 'name': 'test'})


def test_mlflow_logger_skips_non_scalar_config(mlflow_logger):
    with patch('mlflow.log_params') as mock_params:
        mlflow_logger.update_config({'lr': 0.01, 'bad': [1, 2, 3]})
    mock_params.assert_called_once_with({'lr': 0.01})


# -- Tests for _DatamintNNUNetTrainer
def test_save_checkpoint_records_path(bridge, tmp_path):
    ckpt = tmp_path / 'checkpoint_best.pth'; ckpt.touch()
    with patch('mlflow.log_artifact'), \
         patch.object(_DatamintNNUNetTrainer, '_super_save_checkpoint'):
        bridge.save_checkpoint(str(ckpt))
    assert bridge._best_checkpoint_path == ckpt


def test_save_checkpoint_logs_artifact(bridge, tmp_path):
    ckpt = tmp_path / 'checkpoint_best.pth'; ckpt.touch()
    with patch('mlflow.log_artifact') as mock_artifact, \
         patch.object(_DatamintNNUNetTrainer, '_super_save_checkpoint'):
        bridge.save_checkpoint(str(ckpt))
    mock_artifact.assert_called_once_with(str(ckpt), artifact_path='nnunet_checkpoints')


def test_log_validation_summary_logs_per_class_dice(bridge, tmp_path):
    summary = {
        'foreground_mean': 0.85,
        'mean': {'liver': {'Dice': 0.91}, 'tumor': {'Dice': 0.74}},
    }
    val_dir = tmp_path / 'validation'
    val_dir.mkdir(parents=True)
    (val_dir / 'summary.json').write_text(json.dumps(summary))
    bridge.fold = 0
    bridge.output_folder = str(tmp_path)

    with patch('mlflow.log_metric') as mock_log:
        bridge._log_validation_summary()

    calls = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
    assert calls['val/dice_liver'] == pytest.approx(0.91)
    assert calls['val/dice_tumor'] == pytest.approx(0.74)
    assert calls['val/dice_mean'] == pytest.approx(0.85)


def test_version_guard_raises_on_old_nnunet():
    import importlib
    import importlib.metadata as _meta
    import datamint.lightning.trainers.specialized.nnunet._nnunet_trainer_bridge as m
    with patch.object(_meta, 'version', return_value='1.0.0'):
        with pytest.raises(ImportError, match='nnunetv2>=2.4'):
            importlib.reload(m)
