# Datamint Python API

![Build Status](https://github.com/SonanceAI/datamint-python-api/actions/workflows/run_test.yaml/badge.svg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python SDK for interacting with the Datamint platform, providing seamless integration for medical imaging workflows, dataset management, and machine learning experiments.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Setup](#-quick-setup)
- [Documentation](#-documentation)
- [Key Components](#-key-components)
- [Command Line Tools](#️-command-line-tools)
- [Examples](#-examples)
- [Support](#-support)

## 🚀 Features

- **Dataset Management**: Download, upload, and manage medical imaging datasets
- **Annotation Tools**: Create, upload, and manage annotations (segmentations, labels, measurements)
- **Experiment Tracking**: Integrated MLflow support for experiment management
- **PyTorch Lightning Integration**: Streamlined ML workflows with Lightning DataModules and callbacks
- **DICOM Support**: Native handling of DICOM files with anonymization capabilities
- **Multi-format Support**: PNG, JPEG, NIfTI, and other medical imaging formats

See the full documentation at https://sonanceai.github.io/datamint-python-api/

## 📦 Installation

> [!NOTE]
> We recommend using a virtual environment to avoid package conflicts.

### From PyPI

To be released soon

### From Source

```bash
pip install git+https://github.com/SonanceAI/datamint-python-api
```

### Virtual Environment Setup

<details>
<summary>Click to expand virtual environment setup instructions</summary>

We recommend that you install Datamint in a dedicated virtual environment, to avoid conflicting with your system packages.
For instance, create the enviroment once with `python3 -m venv datamint-env` and then activate it whenever you need it with:

1. **Create the environment** (one-time setup):
   ```bash
   python3 -m venv datamint-env
   ```

2. **Activate the environment** (run whenever you need it):
   
   | Platform | Command |
   |----------|---------|
   | Linux/macOS | `source datamint-env/bin/activate` |
   | Windows CMD | `datamint-env\Scripts\activate.bat` |
   | Windows PowerShell | `datamint-env\Scripts\Activate.ps1` |

3. **Install the package**:
   ```bash
   pip install git+https://github.com/SonanceAI/datamint-python-api
   ```

</details>

## Setup API key

To use the Datamint API, you need to setup your API key (ask your administrator if you don't have one). Use one of the following methods to setup your API key:

### Method 1: Command-line tool (recommended)

Run ``datamint-config`` in the terminal and follow the instructions. See [command_line_tools](https://sonanceai.github.io/datamint-python-api/command_line_tools.html) for more details.

### Method 2: Environment variable

Specify the API key as an environment variable.

**Bash:**
```bash
export DATAMINT_API_KEY="my_api_key"
# run your commands (e.g., `datamint-upload`, `python script.py`)
```

**Python:**
```python
import os
os.environ["DATAMINT_API_KEY"] = "my_api_key"
```

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| [🚀 Getting Started](docs/source/getting_started.rst) | Step-by-step setup and basic usage |
| [📖 API Reference](docs/source/client_api.rst) | Complete API documentation |
| [🔥 PyTorch Integration](docs/source/pytorch_integration.rst) | ML workflow integration |
| [💡 Examples](examples/) | Practical usage examples |

## 🔗 Key Components

### Dataset Management

```python
from datamintapi import Dataset

# Load dataset with annotations
dataset = Dataset(
    project_name="medical-segmentation",
)

# Access data
for sample in dataset:
    image = sample['image']       # torch.Tensor
    mask = sample['segmentation'] # torch.Tensor (if available)
    metadata = sample['metainfo'] # dict
```


### PyTorch Lightning Integration

```python
import lightning as L
from datamint.lightning import DatamintDataModule
from datamint.mlflow.lightning.callbacks import MLFlowModelCheckpoint

# Data module
datamodule = DatamintDataModule(
    project_name="your-project",
    batch_size=16,
    train_split=0.8
)

# ML tracking callback
checkpoint_callback = MLFlowModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    register_model_name="best-model"
)

# Trainer with MLflow logging
trainer = L.Trainer(
    max_epochs=100,
    callbacks=[checkpoint_callback],
    logger=L.pytorch.loggers.MLFlowLogger(
        experiment_name="medical-segmentation"
    )
)
```


### Annotation Management


```python
# Upload segmentation masks
api.upload_segmentations(
    resource_id="resource-123",
    file_path="segmentation.nii.gz",
    name="liver_segmentation",
    frame_index=0
)

# Add categorical annotations
api.add_image_category_annotation(
    resource_id="resource-123",
    identifier="diagnosis",
    value="positive"
)

# Add geometric annotations
api.add_line_annotation(
    point1=(10, 20),
    point2=(50, 80),
    resource_id="resource-123",
    identifier="measurement",
    frame_index=5
)
```


## 🛠️ Command Line Tools

### Upload Resources

**Upload DICOM files with anonymization:**
```bash
datamint-upload \
    --path /path/to/dicoms \
    --recursive \
    --channel "training-data" \
    --anonymize \
    --publish
```

**Upload with segmentation masks:**
```bash
datamint-upload \
    --path /path/to/images \
    --segmentation_path /path/to/masks \
    --segmentation_names segmentation_config.yaml
```

### Configuration Management

```bash
# Interactive setup
datamint-config

# Set API key
datamint-config --api-key "your-key"
```

## 🔍 Examples

### Medical Image Segmentation Pipeline

```python
import torch
import lightning as L
from datamint.lightning import DatamintDataModule
from datamint.mlflow.lightning.callbacks import MLFlowModelCheckpoint

class SegmentationModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Model definition...
    
    def training_step(self, batch, batch_idx):
        # Training logic...
        pass

# Setup data
datamodule = DatamintDataModule(
    project_name="liver-segmentation",
    batch_size=8,
    train_split=0.8
)

# Setup model with MLflow tracking
model = SegmentationModel()
checkpoint_cb = MLFlowModelCheckpoint(
    monitor="val_dice",
    mode="max",
    register_model_name="liver-segmentation-model"
)

# Train
trainer = L.Trainer(
    max_epochs=50,
    callbacks=[checkpoint_cb],
    logger=L.pytorch.loggers.MLFlowLogger()
)
trainer.fit(model, datamodule)
```

## 🆘 Support

[Full Documentation](https://datamint-python-api.readthedocs.io/) 
[GitHub Issues](https://github.com/SonanceAI/datamint-python-api/issues)

