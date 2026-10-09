---
name: datamint-end-to-end
description: Plan and run a complete Datamint pipeline with the `datamint` SDK - from raw files to a deployed model - get data in, check annotations, build the dataset and splits, train, evaluate, deploy, run inference and send predictions back for review. Gives the order of stages, what each stage hands to the next, the checks between stages, and which Datamint skill covers each stage. Use this whenever the user wants a full workflow or a script covering several stages (e.g. "upload my scans, train a model and deploy it"), wants to know where to start with Datamint, or is planning a project end to end.
---

# End-to-end Datamint pipeline

This skill is the map. Each stage has its own skill with the details. Read the stage's skill before writing its code.

| Stage | Skill | Core call |
|---|---|---|
| 1. Get data in (files, masks, labels) | `datamint-upload` | `api.resources.upload_resources(..., publish_to=proj)` |
| 2. Check data and annotations | `datamint-data-access` | `api.resources.get_list(project_name=...)`, `api.annotations.get_list(...)` |
| 3. Dataset and splits | `datamint-training` | `build_dataset(project)`, `ds.split(...)`, `parts.save()` |
| 4a. Train a Datamint model | `datamint-training` | `UNetPPTrainer(project=...).fit()` |
| 4b. Or bring an existing model | `datamint-custom-models` | `DatamintModel` adapter + `api.models.log_model(...)` |
| 5. Evaluate and compare | `datamint-experiments` | `Evaluator(dataset=...).evaluate(models=[...])` |
| 6. Deploy and infer | `datamint-deploy` | `api.deploy.start(model_name=...)`, `api.inference.submit(...)` |
| 7. Review predictions | `datamint-deploy` + `datamint-upload` | `save_results=True` with a worklist, annotators correct in the web app |

## What each stage hands to the next

Keep these consistent across the whole pipeline. Most end-to-end failures are a mismatch here.

- **Project name.** Every stage refers to the same project. Create it once with `api.projects.create(name, description, exists_ok=True)`.
- **Label names.** The segment/category names given at upload, the labels in the project, the `class_names` of a model, and the annotation names an adapter returns should be the same strings.
- **Model name.** Trainers register under `model_name=` (default: the project name). Deploy and inference use that same name. Set it explicitly to avoid collisions between models trained on one project.
- **Split.** Persist it (`parts.save()`) before training, so training, evaluation and later runs use the same split. `split_as_of_timestamp` replays an exact earlier one.
- **Model version.** Each registration adds a version. A deploy without version or alias uses the highest version, so register, check, then deploy.

## Checks between stages

| After | Check | If it fails |
|---|---|---|
| 1. Upload | Result list has no `Exception` entries. Resource count in the project matches. With masks: `proj.fetch_worklists()` shows the worklist | `datamint-upload`, "Rules that fail silently" |
| 3. Dataset | `ds.segmentation_labels_set` / `ds.image_labels_set` are not empty and contain the expected labels. `len(parts['train'])` > 0 | `allow_external_annotations=True`, `trusted_annotation_sources`, filters (`datamint-training`) |
| 4. Train | `results['test_results']` has the metric. `api.models.get_by_name(model_name)` finds the model | `datamint-training`, `datamint-experiments` |
| 4b. Adapter | `validate_model(adapter, sample_input=...)` has no `[x]` | `datamint-custom-models` |
| 6. Deploy | `job.status == 'completed'` after `job.wait()` (wait doesn't raise on failure) | `job.error_message`, `job.get_logs()` |
| 6. Inference | `inf.status == 'completed'`, `inf.predictions` has one list per input resource | `inf.get_pod_logs(...)` |

Annotations are often made by people in the web app between stages 1 and 3. A script can't skip that wait: stop after upload and tell the user what to annotate, or start from data that already has masks or labels.

## Minimal script (2D segmentation, masks already available)

```python
from datamint import Api
from datamint.dataset import build_dataset
from datamint.lightning import UNetPPTrainer

api = Api()
PROJECT, MODEL = 'Breast US', 'breast-us-unetpp'
proj = api.projects.create(PROJECT, 'Breast ultrasound lesion segmentation', exists_ok=True)

# 1. upload with masks (datamint-upload)
ids = api.resources.upload_resources(
    image_paths, publish_to=proj, on_error='skip', progress_bar=True,
    segmentation_files=[{'files': [m], 'names': {1: 'Lesion'}} for m in mask_paths],
)
assert not [r for r in ids if isinstance(r, Exception)]

# 3. dataset and persisted split (datamint-training)
ds = build_dataset(PROJECT)
assert 'Lesion' in ds.segmentation_labels_set
ds.split(train=0.7, val=0.15, test=0.15, seed=42).save()

# 4. train and register (datamint-training)
results = UNetPPTrainer(project=PROJECT, image_size=256, max_epochs=30, model_name=MODEL).fit()
print(results['test_results'])

# 6. deploy and infer (datamint-deploy)
job = api.deploy.start(model_name=MODEL)
job.wait(timeout=1800)
assert job.status == 'completed', job.error_message

new = api.resources.get_list(project_name=PROJECT, tags=['batch-2'])   # resources to predict on
inf = api.inference.submit(model_name=MODEL, resource_ids=[r.id for r in new], save_results=True)
inf.wait()
```

Adapt it to the task: classification uses `ImageClassificationTrainer` / `EfficientNetV2Trainer` with category labels, detection uses `YOLOXTrainer` with boxes, and true 3D uses `UNETRPPTrainer` or `NNUNetTrainer`. For an existing model, replace stage 4 with the adapter flow from `datamint-custom-models`.

## Shortcuts

- **No data yet:** `datamint example busi --project MyBusiProject` creates a ready, annotated project (`bccd` detection, `busi` 2D segmentation, `synapse` 3D segmentation, `fracatlas` classification).
- **Project scaffold:** `datamint init` asks for a project name, task (detection, segmentation, classification) and whether to use example data, then generates `01_upload_data.py`, `02_explore.py`, `03_dataset.py`, `04_train.py`, `05_evaluate.py`, `06_deploy.py`. It's interactive, so suggest it to the user rather than running it from an agent.
- **No-code path:** `datamint upload` → annotate in the web app → `datamint train --project P` → `datamint inference file.png --model-name M` (local). Deploying still needs the Python SDK.
- **Full examples:** `notebooks/06_end_to_end/` (classification, 2D segmentation, detection, UNETR++ and nnU-Net 3D, SAM promptable segmentation).
