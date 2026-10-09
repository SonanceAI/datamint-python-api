---
name: datamint-experiments
description: Track, compare and manage Datamint experiments and models with MLflow and the `datamint` SDK - pointing MLflow at the Datamint server, logging params/metrics/models from custom training code, Lightning checkpoint callbacks that register models, finding and comparing runs, the model registry (`api.models`: list, inspect versions, metrics, task type, deployment status, clone to another project, delete), and evaluating/comparing registered or in-memory models on a dataset with `Evaluator`. Use this whenever the user wants to see training results, compare runs or models, pick the best model, log metrics from their own code, inspect or manage registered models, or score models on a Datamint dataset, even if they don't mention MLflow.
---

# Experiments, model registry and evaluation

Datamint hosts the MLflow tracking server and model registry. Built-in trainers (`datamint-training`) and `api.models.log_model` (`datamint-custom-models`) already log and register everything. This skill covers reading, comparing and managing those results, and logging from own code.

## Connecting MLflow to Datamint

```python
import mlflow
from datamint.mlflow import set_project

set_project('MyProject')        # also returns the Project
```

- Importing `datamint.mlflow` configures MLflow to use the Datamint server. Import it before any `mlflow` call, otherwise MLflow uses its own local default instead of Datamint.
- `set_project` sets the **active project**. The model registry is scoped to it, and the project name becomes the default experiment name for runs started without `mlflow.set_experiment`.
- Without `set_project`, the active project comes from the `DATAMINT_PROJECT_ID` or `DATAMINT_PROJECT_NAME` env vars. `set_project` exports `DATAMINT_PROJECT_ID`, so subprocesses inherit it.
- Registry lookups (`api.models.*`) are resolved against the active project. Call `set_project` first when looking for a model in a specific project.

Where results land by default:

| Source | MLflow experiment | Registered model name |
|---|---|---|
| Built-in trainers | `'Training'` (`mlflow_experiment_name=` overrides) | Project name (`model_name=` overrides) |
| `api.models.log_model` | `model_name` | `model_name` |
| `Evaluator.evaluate` | `'Evaluation'` (`experiment_name=` overrides) | Not registered |

## Logging from own training code

Plain MLflow calls work once `datamint.mlflow` is imported and the project is set:

```python
mlflow.set_experiment('nnunet-liver')
with mlflow.start_run(run_name='fold0'):
    mlflow.log_params({'lr': 1e-3, 'batch_size': 8})
    for epoch, m in enumerate(epoch_metrics):
        mlflow.log_metrics(m, step=epoch)       # e.g. {'train/loss': 0.4, 'val/dice': 0.7}
```

To make the result deployable, log a `DatamintModel` adapter instead of a raw network (`datamint-custom-models`).

Lightning: use the MLflow logger plus a checkpoint callback that logs and registers the model:

```python
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from datamint.mlflow.lightning.callbacks import MLFlowModelCheckpoint

ckpt = MLFlowModelCheckpoint(monitor='val/loss', mode='min', model_name='my-unet', register_model_on='test')
trainer = L.Trainer(logger=MLFlowLogger(experiment_name='my-unet'), callbacks=[ckpt], max_epochs=20)
```

- `MLFlowModelCheckpoint` (= `MLFlowPyTorchModelCheckpoint`) logs plain Lightning modules with `mlflow.pytorch`. For modules that are `DatamintModel`s, use `MLFlowDatamintModelCheckpoint`.
- `model_name=None` logs without registering. `register_model_on`: `'train'`, `'val'`, `'test'` (default) or `'predict'`, registering at the end of that stage.
- `log_model_at_end_only=True` (default) logs the model once at the end instead of at every checkpoint. Other options: `code_paths`, `extra_pip_requirements`, `additional_metadata`. Remaining kwargs go to Lightning's `ModelCheckpoint`.

## Finding and comparing runs

Standard MLflow queries work against the Datamint server:

```python
runs = mlflow.search_runs(experiment_names=['Training'], order_by=['metrics.`val/dice` DESC'], max_results=10)
runs[['run_id', 'tags.mlflow.runName', 'metrics.val/dice', 'params.max_epochs']]
```

Metric names with `/` need backticks in `order_by` and `filter_string`. Built-in trainers log metrics with `train/`, `val/` and `test/` prefixes. Check `runs.columns` for the exact names.

## Model registry (`api.models`)

```python
from datamint import Api
api = Api()

models = api.models.get_list()                      # get_list(only_deployed=True) for models with a deployed image
model = api.models.get_by_name('unetpp_r34')        # None if not found
versions = model.get_versions()
v = model.get_latest_version()                       # highest version number; get_latest_version(alias='x') for an alias
```

| Call | Returns |
|---|---|
| `v.version`, `v.run_id`, `v.aliases`, `v.tags`, `v.creation_timestamp` | Version info |
| `v.get_metrics()` | Metrics of the training run (`{}` if there's no run, e.g. a model registered from outside) |
| `v.get_hyperparameters()` | Params of the training run |
| `v.get_task_type()`, `v.get_supported_modes()`, `v.get_annotation_specs()` | What the model predicts (`datamint`-flavor models) |
| `v.load_model(device='cpu')` | The `DatamintModel`, ready for `.predict(resources)` |
| `v.is_deployed()`, `model.is_deployed()` | Whether a deployed image exists |
| `model.get_metrics()`, `model.get_supported_modes()` | Shortcuts for the latest version |
| `model.get_projects()` | Projects the model is associated with |

Picking the best version by a metric:

```python
best = max(model.get_versions(), key=lambda v: v.get_metrics().get('test/dice', float('-inf')))
```

Management:
- `api.models.create(name, description=...)`: register an empty model name (returns the existing one by default, `exists_ok=True`).
- `api.models.clone_model('unetpp_r34', 'OtherProject', version=3)`: copy a version from the active project into another project. Exactly one of `version` / `alias` is required when passing a name. Works only for models logged with the `datamint` flavor (trainers, `DatamintModel` adapters). `target_model_name=` renames it.
- `api.models.delete_model_version(name, version)`, `api.models.delete_registered_model(name)`: irreversible. Confirm with the user first.

Deployment of a registered model: `datamint-deploy` skill.

## Evaluating and comparing models on a dataset

```python
from datamint import ImageDataset
from datamint.evaluate import Evaluator

evaluator = Evaluator(dataset=ImageDataset(project='BUSI_Segmentation'))
results = evaluator.evaluate(models=['unetpp_r34', 'deeplabv3plus'])

for name, r in results.items():
    print(name, r.scores.dataset, r.mlflow_run_id)
```

Each entry in `models` can be:
- a registered model name (latest version),
- a `Model` or `ModelVersion` (specific version; two versions of one model show up as `name_v3`, `name_v5`),
- an in-memory `DatamintModel` that was never registered,
- a `(model, config_dict)` pair, where `config_dict` is passed as `predict()` params and logged as hyperparameters.

`evaluate()` options:
- `prefer_deployed=True`: predict through the deployed serving pod instead of loading the model locally.
- `log_to_mlflow=True` (default): one MLflow run per model in the `'Evaluation'` experiment, with a shared `evaluation_id` tag per call.
- `save_results=True`: write each model's predictions back to the resources as annotations.

`EvaluationResult` fields: `model_name`, `predictions`, `scores`, `hyperparameters`, `mlflow_run_id`. Scores are Dice/IoU for segmentation, accuracy/precision/recall/F1 for classification, and mAP50 and mAP50:95 for detection.

`Benchmark` (`datamint-training`) trains several models on the same split and returns a leaderboard. `Evaluator` scores models that already exist.
