# 06 — End-to-End Use Cases

Complete pipelines from raw data to a deployed model. Each notebook covers data upload, dataset preparation, training, experiment tracking, and deployment.

## Slice-based (2D)

Models that operate on individual 2D images or 2D slices extracted from volumes.

| Notebook | Level | Task | Dataset | Architecture |
|---|---|---|---|---|
| [01_fracatlas_classification](slice_based/01_fracatlas_classification.ipynb) | ![Advanced](https://img.shields.io/badge/level-advanced-red) | Binary classification | FracAtlas (X-rays) | ResNet |
| [02_busi_segmentation](slice_based/02_busi_segmentation.ipynb) | ![Advanced](https://img.shields.io/badge/level-advanced-red) | Semantic segmentation | BUSI (ultrasound) | TransUNet |
| [03_bccd_detection](slice_based/03_bccd_detection.ipynb) | ![Advanced](https://img.shields.io/badge/level-advanced-red) | Object detection | BCCD (blood cells) | YOLOX |

## Full 3D

Models that consume entire volumetric inputs without slicing.

| Notebook | Level | Task | Dataset | Architecture |
|---|---|---|---|---|
| [01_synapse_unetrpp](full_3d/01_synapse_unetrpp.ipynb) | ![Advanced](https://img.shields.io/badge/level-advanced-red) | Multi-organ segmentation | Synapse CT | UNETR++ |
| [02_synapse_nnunet](full_3d/02_synapse_nnunet.ipynb) | ![Advanced](https://img.shields.io/badge/level-advanced-red) | Multi-organ segmentation | Synapse CT | nnU-Net |
