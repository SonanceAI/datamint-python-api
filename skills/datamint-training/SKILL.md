---
name: datamint-training
description: Train Datamint models on Datamint projects with the `datamint` SDK - the built-in one-line trainers (UNet++, DeepLabV3+, TransUNet, UNETR++, nnU-Net, EfficientNetV2, timm classifiers, YOLOX), a custom architecture as a `SegmentationModule` / `ClassificationModule` subclass, PyTorch datasets (`ImageDataset`, `VolumeDataset`, sliced volumes, videos), item and mask formats, train/val/test and patient-wise splits, benchmarking several models, resuming paused runs, and the `datamint train` CLI. Use this whenever the user wants to train, fine-tune or compare segmentation, classification or detection models on Datamint data, or load a Datamint project as a PyTorch dataset, even if they only say "train a model on my project".
---

# Training on Datamint data

Start at the highest level that fits:

| Need | Use |
|---|---|
| Train a standard model, no code | `datamint train --project P` |
| Standard model from Python, MLflow logging and registration included | One-line trainer, e.g. `UNetPPTrainer(project=P).fit()` |
| Own architecture, keep Datamint loss/metrics/MLflow/deploy | Generic trainer + `SegmentationModule` / `ClassificationModule` subclass |

All three produce a Datamint model, ready to deploy. For a model the user already has, a plain `LightningModule`, or their own training loop (`DatamintDataModule`), use the `datamint-custom-models` skill. The dataset and split sections below apply there too.

Trainers log metrics and checkpoints to MLflow and register the model. For reading runs and the registry afterwards, see the `datamint-experiments` skill. For deployment, see `datamint-deploy`.

## Built-in trainers

```python
from datamint.lightning import UNetPPTrainer

trainer = UNetPPTrainer(project='BUSI_Segmentation', image_size=256, batch_size=16, max_epochs=20, accelerator='auto')
results = trainer.fit()          # {'model': ..., 'test_results': ...}
```

| Trainer | Task | Data |
|---|---|---|
| `UNetPPTrainer` | 2D segmentation, UNet++ (`resnet34`) | 2D images; slices 3D volume projects automatically |
| `DeepLabV3PlusTrainer` | 2D segmentation | Same as above |
| `TransUNetTrainer` | 2D segmentation, fixed 224x224 | Same as above |
| `UNETRPPTrainer` | 3D segmentation, UNETR++ | 3D volumes only, no slicing |
| `NNUNetTrainer` | 2D/3D segmentation, nnU-Net v2 | 3D volume projects; `configuration='2d' / '3d_fullres' / '3d_lowres' / '3d_cascade_fullres'` |
| `ImageClassificationTrainer` | 2D classification, `timm` backbone | 2D images |
| `EfficientNetV2Trainer` | 2D classification | 2D images |
| `YOLOXTrainer` | 2D detection | 2D images |

Common constructor arguments (all trainers except where noted):
- exactly one of `project=` or `dataset=` (a prebuilt dataset)
- `batch_size` (16), `num_workers` (4), `max_epochs` (1), `early_stopping_patience` (10, `None` disables)
- `train_transform`, `eval_transform` (albumentations) override the task defaults
- `model_name` (registry name, auto-generated if omitted), `mlflow_experiment_name`, `run_name`
- `split_as_of_timestamp` to reuse an exact historical split
- `resume_from` = MLflow `run_id` or checkpoint path printed when a run was interrupted
- anything else (`accelerator`, `devices`, `precision='16-mixed'`, ...) goes to `lightning.Trainer`, as does `trainer_kwargs={...}`

After `fit()`: `trainer.dataset`, `trainer.datamodule`, `trainer.model`. Evaluation only: `trainer.test(register_model=False)`.

Notes:
- The default `max_epochs=1` is a smoke-test value. Set it explicitly for real training.
- Projects with images of different sizes need `image_size=` (or a resize transform), otherwise batching fails with "mismatched shapes".
- 2D segmentation trainers on volume projects pick the slice plane from spacing and shape, falling back to axial. Override with `slice_axis='coronal'`.
- `NNUNetTrainer` needs `pip install datamint[nnunet]`, runs nnU-Net's own pipeline (long `fit()`), doesn't accept `model=`, and resumes with `continue_training=True` instead of `resume_from`.

## Custom architecture inside a trainer

Subclass `SegmentationModule` or `ClassificationModule` and pass the **class** (`model=MAnetModule`, not `MAnetModule()`). The trainer instantiates it and injects the default `loss_fn` and `metrics_factories`, and the result stays deployable as a Datamint model.

```python
import segmentation_models_pytorch as smp
from datamint.lightning import SemanticSegmentation2DTrainer
from datamint.lightning.trainers.lightning_modules import SegmentationModule

class MAnetModule(SegmentationModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, class_names=['benign', 'malignant'], **kwargs)
        self.model = smp.MAnet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=2)

    def forward(self, x):
        return self.model(x)

results = SemanticSegmentation2DTrainer(project='BUSI_Segmentation', image_size=256, model=MAnetModule).fit()
```

Generic trainers: `SemanticSegmentation2DTrainer` (2D, or slices volumes; `slice_axis=`), `SemanticSegmentation3DTrainer` (slice-based on volume projects), `VolumeSegmentationTrainer` (true 3D), `ClassificationTrainer`, `DetectionTrainer`.

## Datasets

```python
from datamint import ImageDataset, VolumeDataset

ds = ImageDataset(project='BUSI_Segmentation')        # 2D: X-ray, PNG/JPEG, single-frame DICOM
vol = VolumeDataset(project='Liver CT')               # 3D: NIfTI, DICOM series
sliced = vol.slice(axis='axial')                      # 2D items from volumes
```

Videos: `datamint.dataset.VideoDataset(project=...)`, and `.frame_by_frame()` for 2D frames.

When the data type isn't known in advance, let the SDK pick the class:

```python
from datamint.dataset import build_dataset

ds = build_dataset('MyProject', include_unannotated=False)   # kwargs go to the dataset constructor
```

`build_dataset` looks at up to 5 resources of the project and returns an `ImageDataset` (2D images), `VolumeDataset` (NIfTI, DICOM, volumes) or `VideoDataset`. It raises `ValueError` for an empty project, unsupported resource types, or mixed types. In those cases, instantiate the class directly.

Each item is a dict:

| Key | Content |
|---|---|
| `image` | Tensor `(C, H, W)` for 2D, `(C, D, H, W)` for volumes |
| `masks` | Segmentations (see below), when `return_segmentations=True` (default) |
| `mask_labels` | Label names, when masks are not in semantic format |
| `boxes`, `box_labels` | When `return_boxes=True` |
| `image_labels`, `image_categories` | Image-level labels and categories (classification targets) |
| `metainfo`, `resource`, `annotations` | Source data |

The mask key is `masks`.

Masks format:
- Default: dict `{annotator: tensor (num_instances, H, W)}` with names in `mask_labels`.
- `return_as_semantic_segmentation=True`: dict `{annotator: tensor (num_labels + 1, H, W)}`.
- Semantic + `semantic_seg_merge_strategy` (`'union'`, `'intersection'`, `'mode'`): one tensor `(num_labels + 1, H, W)`. This is what models usually need. **Background is channel 0**, so built-in modules use `batch['masks'][:, 1:]`.

Useful constructor options:

| Option | Effect |
|---|---|
| `include_unannotated` (default `True`) | Set `False` to drop resources without annotations |
| `include_segmentation_names` / `exclude_segmentation_names` | Keep or drop segment labels |
| `include_image_label_names`, `include_frame_label_names` (+ `exclude_`) | Same for labels |
| `include_annotators` / `exclude_annotators` | Filter by annotator email |
| `alb_transform` | Albumentations transform applied to image and targets together |
| `trusted_annotation_sources` (default `('imported', 'manual')`) | Excludes model predictions (`model_deploy`, `model_pipeline`) so a model isn't retrained on its own outputs. `None` keeps all sources |
| `allow_external_annotations` (default `False`) | Include annotation labels that are not part of the project's official schema (e.g. labels from other projects or legacy annotations) |
| `resources=` | Build from a list of resources instead of `project=` |

**Items come back with no annotations (empty masks, empty `image_labels`, empty `*_labels_set`)?** Most likely the annotations use labels that are not in the project's schema, and `allow_external_annotations=False` filters them out. Rebuild with `allow_external_annotations=True` (also for `build_dataset` and `dataset_kwargs=` of trainers). If that doesn't help, check `trusted_annotation_sources` (model predictions are excluded by default) and the `include_*` / `exclude_*` filters.

Dataset methods: `filter(tags=..., filename_pattern=..., has_annotations=..., annotation_names=..., custom_fn=...)` (chainable), `subset(indices)`, `group_by_patient()`, `segmentation_labels_set`, `image_labels_set`, `prefetch()`.

DataLoader: use `ds.get_dataloader(batch_size=..., shuffle=..., num_workers=...)`, which applies the dataset's collate function. For detection, use `detection_collate_fn` from `datamint.dataset` (boxes stay as lists).

## Splits

```python
parts = ds.split()                                               # project split assignments
parts = ds.split(train=0.7, val=0.15, test=0.15, seed=42)        # local random split
parts = ds.split(train=0.8, test=0.2, by_patient=True, seed=42)  # whole patients per split, no leakage
parts.save()                                                     # persist assignments to the project
train_ds, snapshot = parts['train'], parts['train'].split_as_of_timestamp
```

- With no ratios on a project dataset, the project's split assignments are used. If the project has none, a local 70/15/15 split is generated with a warning and **not saved**. Call `parts.save()` to make it reproducible.
- `by_patient=True` needs ratios. `none_patient_id_strategy` handles resources without a patient id: `'individual'` (default), `'group'`, `'skip'`, `'error'`.
- Pass `split_as_of_timestamp` to a trainer to replay the exact same split later. It's also logged to MLflow.
- `Benchmark` requires existing split assignments.

## Comparing models

```python
from datamint.lightning import Benchmark, UNetPPTrainer, DeepLabV3PlusTrainer

bench = Benchmark(
    dataset=ImageDataset(project='BUSI_Segmentation', return_segmentations=True),
    trainers=[(UNetPPTrainer, {'model_name': 'unetpp_r34'}),
              (DeepLabV3PlusTrainer, {'model_name': 'deeplabv3plus'})],
    main_metric='dice', main_metric_mode='max', max_epochs=20,
)
leaderboard = bench.run()     # pandas DataFrame: model_name, val/test metric, run_id, registered_version
```

Every trainer needs a unique `model_name`, and all trainers must be from the same task family. `bench.save_config('b.yaml')` / `Benchmark.load_from_file('b.yaml', dataset=...)` re-run it later.

## CLI

```bash
datamint train --project MyProject --dry-run              # show detected task, format, model, settings
datamint train --project MyProject --model unetpp --max-epochs 20 --image-size 256
datamint train --project MyProject --model yolox --resume <run_id>
datamint train --interactive
```

`--model`: `unetpp`, `deeplabv3plus`, `transunet`, `unetrpp` (3D), `nnunet` (3D), `classification`, `efficientnetv2`, `yolox`. Task and 2D/3D format are auto-detected from the project. Advanced options (losses, transforms, custom models) need the Python SDK.

No data yet: `datamint example busi --project MyBusiProject` creates a ready project (`bccd` detection, `busi` 2D segmentation, `synapse` 3D segmentation, `fracatlas` classification).
