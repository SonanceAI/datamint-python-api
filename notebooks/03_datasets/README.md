# 03 — Datasets

PyTorch dataset classes, data splits, and volume loading.

| Notebook | Description |
|---|---|
| [01_project_scoped_splits](01_project_scoped_splits.ipynb) | Create reproducible train/val/test splits and persist them to the server |
| [02_patient_wise_splits](02_patient_wise_splits.ipynb) | Patient-level splitting to prevent data leakage in multi-scan datasets |
| [03_build_dataset](03_build_dataset.ipynb) | Use `build_dataset` to auto-detect project type and get the right dataset class |
| [04_volume_dataset](04_volume_dataset.ipynb) | Load 3D volumes, slice along anatomical axes, and apply albumentations transforms |
