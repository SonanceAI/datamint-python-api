"""
Test nnU-Net training pipeline using NNUNetTrainer. We mock the dataset building, nnUNet environment setup, fingerprinting/planning, preprocessing, training, prediction, and import steps to verify that they are
called in the correct order during fit(). We also test the dataset ID assignment logic and that the expected files are written during fingerprinting/planning.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
import datamint.lightning.trainers.specialized.nnunet.trainer as _trainer_mod
from datamint.lightning.trainers.specialized.nnunet.trainer import NNUNetTrainer

# Fake nnunetv2 submodules so patch() can resolve them when the trainer imports them.
def _mock_nnunetv2():
    root = MagicMock()
    root.__version__ = '2.4.0'
    submodules = {
        'nnunetv2': root,
        'nnunetv2.experiment_planning': MagicMock(),
        'nnunetv2.experiment_planning.dataset_fingerprint': MagicMock(),
        'nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor': MagicMock(),
        'nnunetv2.experiment_planning.experiment_planners': MagicMock(),
        'nnunetv2.experiment_planning.experiment_planners.default_experiment_planner': MagicMock(),
        'nnunetv2.experiment_planning.plan_and_preprocess_api': MagicMock(),
        'nnunetv2.training': MagicMock(),
        'nnunetv2.training.nnUNetTrainer': MagicMock(),
        'nnunetv2.training.nnUNetTrainer.nnUNetTrainer': MagicMock(),
    }

    class _FakeNNUNetTrainer:
        def log(self, *a, **kw): pass
        def save_checkpoint(self, *a, **kw): pass
        def perform_actual_validation(self, *a, **kw): pass
        def print_to_log_file(self, *a, **kw): pass
    submodules['nnunetv2.training.nnUNetTrainer.nnUNetTrainer'].nnUNetTrainer = _FakeNNUNetTrainer
    for name, mod in submodules.items():
        sys.modules.setdefault(name, mod)

_mock_nnunetv2()

@pytest.fixture()
def trainer(tmp_path):
    mock_dataset = MagicMock(
        resources=[MagicMock(id='res-001'), MagicMock(id='res-002')],
        seglabel_list=['liver', 'tumor'],
    )
    with patch.object(NNUNetTrainer, '_build_dataset', return_value=mock_dataset):
        t = NNUNetTrainer(
            project='CT_Liver',
            configuration='3d_fullres',
            fold=0,
            nnunet_work_dir=tmp_path,
        )
        yield t


# -- Dataset ID assignment tests
def test_assign_dataset_id_increments(tmp_path):
    registry = tmp_path / 'nnunet_dataset_ids.yaml'
    registry.write_text('CT_Brain: 1\n')
    with patch.object(_trainer_mod, 'REGISTRY_PATH', registry):
        t1 = NNUNetTrainer(project='CT_Liver', nnunet_work_dir=tmp_path)
        t2 = NNUNetTrainer(project='CT_Kidney', nnunet_work_dir=tmp_path)
        id1 = t1._assign_dataset_id()
        id2 = t2._assign_dataset_id()
    assert id1 == 2 and id2 == 3 and id1 != id2


def test_assign_dataset_id_is_stable_for_same_project(tmp_path):
    registry = tmp_path / 'nnunet_dataset_ids.yaml'
    with patch.object(_trainer_mod, 'REGISTRY_PATH', registry):
        t = NNUNetTrainer(project='CT_Liver', nnunet_work_dir=tmp_path)
        assert t._assign_dataset_id() == t._assign_dataset_id()


# -- nnUNet environment variable setup tests
def test_set_nnunet_env_sets_all_three_vars(trainer, tmp_path, monkeypatch):
    monkeypatch.delenv('nnUNet_raw', raising=False)
    monkeypatch.delenv('nnUNet_preprocessed', raising=False)
    monkeypatch.delenv('nnUNet_results', raising=False)
    trainer._set_nnunet_env()
    assert all(k in os.environ for k in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'))
    assert os.environ['nnUNet_raw'].startswith(str(tmp_path))


# -- nnUNet fingerprinting and planning tests
def test_fingerprint_and_plan_writes_expected_files(trainer, tmp_path):
    preprocessed_dir = tmp_path / 'preprocessed' / 'Dataset001_CTLiver'
    preprocessed_dir.mkdir(parents=True)
    trainer._set_nnunet_env()

    def fake_fp():
        (preprocessed_dir / 'dataset_fingerprint.json').write_text('{}')

    def fake_plan():
        (preprocessed_dir / 'nnUNetPlans.json').write_text('{}')

    with patch('nnunetv2.experiment_planning.dataset_fingerprint'
               '.fingerprint_extractor.DatasetFingerprintExtractor') as MockFP, \
         patch('nnunetv2.experiment_planning.experiment_planners.default_experiment_planner.ExperimentPlanner') as MockPlan:
        MockFP.return_value.run.side_effect = fake_fp
        MockPlan.return_value.plan_experiment.side_effect = fake_plan
        trainer._run_fingerprint_and_plan(dataset_id=1)

    assert (preprocessed_dir / 'dataset_fingerprint.json').exists()
    assert (preprocessed_dir / 'nnUNetPlans.json').exists()


def test_fingerprint_raises_if_json_not_written(trainer):
    with patch('nnunetv2.experiment_planning.dataset_fingerprint'
               '.fingerprint_extractor.DatasetFingerprintExtractor') as MockFP, \
         patch('nnunetv2.experiment_planning.experiment_planners.default_experiment_planner.ExperimentPlanner'):
        MockFP.return_value.run.return_value = None
        with pytest.raises(RuntimeError, match='dataset_fingerprint.json'):
            trainer._run_fingerprint_and_plan(dataset_id=1)


# -- Full pipeline execution order test
def test_fit_calls_pipeline_steps_in_order(trainer):
    order = []
    with patch.object(trainer, '_assign_dataset_id', side_effect=lambda: order.append('id') or 1), \
         patch.object(_trainer_mod, 'DatamintToNNUNetExporter'), \
         patch.object(trainer, '_set_nnunet_env',           side_effect=lambda: order.append('env')), \
         patch.object(trainer, '_run_fingerprint_and_plan', side_effect=lambda _: order.append('fp')), \
         patch.object(trainer, '_run_preprocessing',        side_effect=lambda _: order.append('pre')), \
         patch.object(trainer, '_build_nnunet_trainer',
                      return_value=MagicMock(run_training=lambda: order.append('train'))), \
         patch.object(trainer, '_run_prediction',       side_effect=lambda *_: order.append('pred')), \
         patch.object(trainer, '_import_predictions',   side_effect=lambda *_: order.append('imp')), \
         patch.object(trainer, '_build_deploy_adapter', side_effect=lambda *_: order.append('deploy')), \
         patch('datamint.lightning.trainers.specialized.nnunet.trainer.mlflow.register_model',
               side_effect=lambda *_: order.append('register')), \
         patch.object(trainer, '_start_mlflow_run'):
        trainer.fit()
    assert order == ['id', 'env', 'fp', 'pre', 'train', 'pred', 'imp', 'deploy', 'register']
