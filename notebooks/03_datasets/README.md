# 03 — Datasets

PyTorch dataset classes, data splits, and volume loading.

| Notebook | Level | Description |
|---|---|---|
| [01_project_scoped_splits](01_project_scoped_splits.ipynb) | ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) | Create reproducible train/val/test splits and persist them to the server |
| [02_patient_wise_splits](02_patient_wise_splits.ipynb) | ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) | Patient-level splitting to prevent data leakage in multi-scan datasets |
| [03_build_dataset](03_build_dataset.ipynb) | ![Intermediate](https://img.shields.io/badge/level-intermediate-yellow) | Use `build_dataset` to auto-detect project type and get the right dataset class |
| [04_volume_dataset](04_volume_dataset.ipynb) | ![Advanced](https://img.shields.io/badge/level-advanced-red) | Load 3D volumes, slice along anatomical axes, and apply albumentations transforms |
