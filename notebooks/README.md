# Datamint Notebooks

Tutorials and end-to-end examples for the Datamint.
Folders are numbered in the recommended learning order.

**Difficulty levels:**

- ![Beginner](https://img.shields.io/badge/level-beginner-brightgreen) no ML knowledge needed
- ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) assumes SDK familiarity, introduces ML/dataset concepts
- ![Advanced](https://img.shields.io/badge/level-advanced-red) full training pipelines, custom models, 3D data, multi-step workflows

## Structure

| Folder | Level | What you'll learn |
|---|---|---|
| [01_getting_started](01_getting_started/) | ![Beginner](https://img.shields.io/badge/level-beginner-brightgreen) | Upload data and explore a project |
| [02_annotations](02_annotations/) | ![Beginner](https://img.shields.io/badge/level-beginner-brightgreen) | Upload and work with annotations |
| [03_datasets](03_datasets/) | ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) | Build PyTorch datasets, splits, and volume loading |
| [04_experiment_tracking](04_experiment_tracking/) | ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) | Log metrics and artifacts with MLflow |
| [05_deployment](05_deployment/) | ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) | Deploy registered and external models |
| [06_end_to_end](06_end_to_end/) | ![Advanced](https://img.shields.io/badge/level-advanced-red) | Full pipelines from data to deployed model |

## Notebooks

### 01 — Getting Started
1. [`01_upload_data`](01_getting_started/01_upload_data.ipynb) ![Beginner](https://img.shields.io/badge/level-beginner-brightgreen) — Upload images and volumes to a project
2. [`02_explore_data`](01_getting_started/02_explore_data.ipynb) ![Beginner](https://img.shields.io/badge/level-beginner-brightgreen) — Browse resources, metadata, and annotations

### 02 — Annotations
1. [`01_upload_annotations`](02_annotations/01_upload_annotations.ipynb) ![Beginner](https://img.shields.io/badge/level-beginner-brightgreen) — Upload segmentation masks and labels
2. [`02_geometry_annotations`](02_annotations/02_geometry_annotations.ipynb) ![Beginner](https://img.shields.io/badge/level-beginner-brightgreen) — Work with boxes, points, lines, and polygon annotations

### 03 — Datasets
1. [`01_project_scoped_splits`](03_datasets/01_project_scoped_splits.ipynb) ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) — Create and persist train/val/test splits on the server
2. [`02_patient_wise_splits`](03_datasets/02_patient_wise_splits.ipynb) ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) — Avoid data leakage with patient-level splitting
3. [`03_build_dataset`](03_datasets/03_build_dataset.ipynb) ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) — Auto-detect dataset type with `build_dataset`
4. [`04_volume_dataset`](03_datasets/04_volume_dataset.ipynb) ![Advanced](https://img.shields.io/badge/level-advanced-red) — Load 3D volumes, slice into 2D, apply albumentations

### 04 — Experiment Tracking
1. [`01_mlflow_manual_logging`](04_experiment_tracking/01_mlflow_manual_logging.ipynb) ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) — Log metrics, parameters, and models manually with MLflow

### 05 — Deployment
1. [`01_deploy_registered_model`](05_deployment/01_deploy_registered_model.ipynb) ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) — Deploy a model already registered in Datamint
2. [`02_deploy_external_model`](05_deployment/02_deploy_external_model.ipynb) ![Advanced](https://img.shields.io/badge/level-advanced-red) — Wrap and deploy a model trained outside Datamint
3. [`03_validate_model`](05_deployment/03_validate_model.ipynb) ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) — Validate a model before promoting it to production
4. [`04_predict_images_volumes_and_videos`](05_deployment/04_predict_images_volumes_and_videos.ipynb) ![Advanced](https://img.shields.io/badge/level-advanced-red) — Run a 2D model on images, 3D volumes, and video frames using automatic prediction bridges

### 06 — End-to-End Use Cases

#### Slice-based (2D)
1. [`01_fracatlas_classification`](06_end_to_end/slice_based/01_fracatlas_classification.ipynb) ![Advanced](https://img.shields.io/badge/level-advanced-red) — Binary fracture classification on X-rays (FracAtlas)
2. [`02_busi_segmentation`](06_end_to_end/slice_based/02_busi_segmentation.ipynb) ![Advanced](https://img.shields.io/badge/level-advanced-red) — 2D breast ultrasound segmentation (BUSI)
3. [`03_bccd_detection`](06_end_to_end/slice_based/03_bccd_detection.ipynb) ![Advanced](https://img.shields.io/badge/level-advanced-red) — Object detection on blood cell images (BCCD) with YOLOX

#### Full 3D
1. [`01_synapse_unetrpp`](06_end_to_end/full_3d/01_synapse_unetrpp.ipynb) ![Advanced](https://img.shields.io/badge/level-advanced-red) — 3D multi-organ segmentation with UNETR++ (Synapse)
2. [`02_synapse_nnunet`](06_end_to_end/full_3d/02_synapse_nnunet.ipynb) ![Advanced](https://img.shields.io/badge/level-advanced-red) — 3D multi-organ segmentation with nnU-Net (Synapse)
