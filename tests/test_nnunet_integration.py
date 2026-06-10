"""
End-to-end integration test for the full nnUNet pipeline.

Run with:
    pytest tests/test_nnunet.py -v -s

Requirements:
    - pip install ".[nnunet]"
    - A real Datamint project named PROJECT_NAME with NIfTI volumes + segmentations
    - Enough disk space for nnUNet preprocessing (~2–5 GB for a small dataset)
"""
import random
import pytest
from pathlib import Path

PROJECT_NAME = "NNUNET__TEST_Synapse_Tutorial"
WORK_DIR = Path.home() / '.cache' / 'datamint' / 'nnunet_integration_test'


@pytest.mark.integration
def test_full_nnunet_pipeline():
    from datamint import Api
    from datamint.lightning import NNUNetTrainer
    import mlflow

    api = Api()

    proj = api.projects.create(
        name=PROJECT_NAME,
        description="Testing nnunet on Synapse Multi-Organ CT",
        exists_ok=True,
    )

    # -- Assign train/test splits randomly (nnUNet requires splits to be defined) --
    all_resources = list(api.resources.get_list(project_name=PROJECT_NAME))
    all_resources.sort(key=lambda r: r.filename)

    random.seed(42)
    random.shuffle(all_resources)

    n = len(all_resources)
    n_train = int(0.70 * n)
    n_test = int(0.30 * n)

    api.projects.assign_splits(proj, all_resources[:n_train], split_name='train')
    api.projects.assign_splits(proj, all_resources[n_train:n_train + n_test], split_name='test')

    print(f"Train: {n_train} Test: {n_test}")

    trainer = NNUNetTrainer(
        project=PROJECT_NAME,
        configuration='2d',
        fold=0,
        max_epochs=2,
        nnunet_work_dir=WORK_DIR,
        continue_training=True
    )

    results = trainer.fit()

    # Testing the full pipeline is a bit tricky since there are many steps and potential failure points.
    # Here we check for key outputs at each stage to verify that the pipeline ran end-to-end:
    # 1. The trainer's fit() method returned a 'bridge' object, which indicates that the deploy adapter ran and 
    # produced an output folder.
    assert 'bridge' in results
    bridge = results['bridge']
    assert bridge is not None

    # 2. The exporter wrote the expected dataset structure for nnUNet, including dataset.json, imagesTr/, labelsTr/,
    # and datamint_case_map.json.
    dataset_dir = WORK_DIR / 'raw' / f'Dataset001_{trainer._dataset_name}'
    assert (dataset_dir / 'dataset.json').exists(), "dataset.json not written by exporter"
    assert (dataset_dir / 'imagesTr').exists(), "imagesTr directory missing"
    assert (dataset_dir / 'labelsTr').exists(), "labelsTr directory missing"
    assert (dataset_dir / 'datamint_case_map.json').exists(), "case map not written"

    # 3. The fingerprinting and planning steps ran and produced dataset_fingerprint.json and nnUNetPlans.json 
    # in the preprocessed directory.
    preprocessed_dir = WORK_DIR / 'preprocessed' / f'Dataset001_{trainer._dataset_name}'
    assert (preprocessed_dir / 'dataset_fingerprint.json').exists(), \
        "dataset_fingerprint.json missing — fingerprinting failed"
    assert (preprocessed_dir / 'nnUNetPlans.json').exists(), \
        "nnUNetPlans.json missing — experiment planning failed"

    # 4. The training step ran and produced a checkpoint_final.pth in the trainer's output folder.
    final_ckpt = Path(bridge.output_folder) / 'checkpoint_final.pth'
    assert final_ckpt.exists(), \
        f"checkpoint_final.pth not found at {final_ckpt} — training did not complete"

    # 5. The deploy adapter registered the model in MLflow and returned the model name. We check that the model exists in the registry.
    assert 'model_name' in results, "fit() did not return model_name, deploy adapter may have failed"
    model_name = results['model_name']
    client = mlflow.MlflowClient()
    registered = client.get_registered_model(model_name)
    assert registered is not None, f"Model '{model_name}' not found in MLflow registry"

    # 6. Finally, we check that the predictions were imported back into Datamint as volume annotations.
    # We look for any resources in the project that have segmentation annotations,
    # if none are found, it's a strong signal that the import step failed.
    resources = list(api.resources.get_list(project_name=PROJECT_NAME))
    assert len(resources) > 0, "No resources found in project"

    annotated = [r for r in resources if r.fetch_annotations(annotation_type='segmentation')]
    assert len(annotated) > 0, \
        "No segmentation annotations found after import, _import_predictions may have failed"

