# Datamint Notebooks

Tutorials and end-to-end examples for the Datamint.
Folders are numbered in the recommended learning order.

## Structure

| Folder | What you'll learn |
|---|---|
| [01_getting_started](01_getting_started/) | Upload data and explore a project |
| [02_annotations](02_annotations/) | Upload and work with annotations |
| [03_datasets](03_datasets/) | Build PyTorch datasets, splits, and volume loading |
| [04_experiment_tracking](04_experiment_tracking/) | Log metrics and artifacts with MLflow |
| [05_deployment](05_deployment/) | Deploy registered and external models |
| [06_end_to_end](06_end_to_end/) | Full pipelines from data to deployed model |

## Notebooks

### 01 — Getting Started
1. [`01_upload_data`](01_getting_started/01_upload_data.ipynb) — Upload images and volumes to a project
2. [`02_explore_data`](01_getting_started/02_explore_data.ipynb) — Browse resources, metadata, and annotations

### 02 — Annotations
1. [`01_upload_annotations`](02_annotations/01_upload_annotations.ipynb) — Upload segmentation masks and labels
2. [`02_geometry_annotations`](02_annotations/02_geometry_annotations.ipynb) — Work with boxes, points, lines, and polygon annotations

### 03 — Datasets
1. [`01_project_scoped_splits`](03_datasets/01_project_scoped_splits.ipynb) — Create and persist train/val/test splits on the server
2. [`02_patient_wise_splits`](03_datasets/02_patient_wise_splits.ipynb) — Avoid data leakage with patient-level splitting
3. [`03_build_dataset`](03_datasets/03_build_dataset.ipynb) — Auto-detect dataset type with `build_dataset`
4. [`04_volume_dataset`](03_datasets/04_volume_dataset.ipynb) — Load 3D volumes, slice into 2D, apply albumentations

### 04 — Experiment Tracking
1. [`01_mlflow_manual_logging`](04_experiment_tracking/01_mlflow_manual_logging.ipynb) — Log metrics, parameters, and models manually with MLflow

### 05 — Deployment
1. [`01_deploy_registered_model`](05_deployment/01_deploy_registered_model.ipynb) — Deploy a model already registered in Datamint
2. [`02_deploy_external_model`](05_deployment/02_deploy_external_model.ipynb) — Wrap and deploy a model trained outside Datamint

### 06 — End-to-End Use Cases

#### Slice-based (2D)
1. [`01_fracatlas_classification`](06_end_to_end/slice_based/01_fracatlas_classification.ipynb) — Binary fracture classification on X-rays (FracAtlas)
2. [`02_busi_segmentation`](06_end_to_end/slice_based/02_busi_segmentation.ipynb) — 2D breast ultrasound segmentation (BUSI)
3. [`03_bccd_detection`](06_end_to_end/slice_based/03_bccd_detection.ipynb) — Object detection on blood cell images (BCCD) with YOLOX

#### Full 3D
1. [`01_synapse_unetrpp`](06_end_to_end/full_3d/01_synapse_unetrpp.ipynb) — 3D multi-organ segmentation with UNETR++ (Synapse)
2. [`02_synapse_nnunet`](06_end_to_end/full_3d/02_synapse_nnunet.ipynb) — 3D multi-organ segmentation with nnU-Net (Synapse)
