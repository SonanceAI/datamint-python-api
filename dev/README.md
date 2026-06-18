# Developer Guide

## Purpose

Quick reference for developers working on the `datamint-python-api` SDK.

## Project Overview

`datamint` is a Python SDK for the Datamint platform, providing:

- **API client** (`datamint.api`) -- REST endpoints for projects, resources, annotations, models, etc.
- **Dataset classes** (`datamint.dataset`) -- modality-aware datasets (image, video, volume).
- **PyTorch Lightning integration** (`datamint.lightning`) -- `DataModule`, trainers, and Lightning modules for classification and segmentation.
- **MLflow flavors** (`datamint.mlflow`) -- custom tracking store, artifact repository, and model flavors.

## Configuration

The SDK uses a local configuration file for authentication and settings. When the user runs [`datamint-config`](./datamint/client_cmd_tools/datamint_config.py), configuration is stored in `~/.config/datamintapi/datamintapi.yaml` (via [PlatformDirs](https://pypi.org/project/platformdirs/)).

## Module Structure

```
datamint/
├── api/                    # REST API client
│   ├── client.py           # Main Api class (entry point)
│   ├── base_api.py         # BaseApi, ApiConfig
│   ├── endpoints/          # Endpoint handlers (projects, resources, etc.)
│   └── dto/                # Data transfer objects
├── dataset/                # Dataset abstractions
│   ├── base.py             # DatamintBaseDataset
│   ├── image_dataset.py    # ImageDataset
│   ├── volume_dataset.py   # VolumeDataset
│   └── video_dataset.py    # VideoDataset
├── entities/               # Domain models
│   ├── annotations/        # Annotation types (box, segmentation, etc.)
│   └── resources/          # Resource types (DICOM, NIfTI, etc.)
├── lightning/              # PyTorch Lightning integration
│   ├── datamodule.py       # DatamintDataModule
│   └── trainers/           # Trainers and Lightning modules
├── mlflow/                 # MLflow integration
│   ├── flavors/            # Model flavors and prediction routing
│   ├── tracking/           # Custom DatamintStore
│   └── artifact/           # DatamintArtifactsRepository
├── client_cmd_tools/       # CLI tools (datamint-upload, datamint-config)
└── utils/                  # Utilities (collection, nifti, visualization)
```

## Development Setup

### With Poetry

```bash
# Install with dev dependencies
poetry install

# Activate environment
poetry shell
```

### Without Poetry (pip)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install from source (editable mode)
pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests
pytest

# Single file
pytest tests/test_api_handler.py -v
```

Test fixtures are defined in [`tests/conftest.py`](../tests/conftest.py) with mock UUIDs and sample resource/user payloads.

## Development tools

### Releasing

Use the release script in [`dev/publish_release.sh`](./publish_release.sh) (requires `gh` CLI tool):

```bash
# Dry run (preview only)
bash dev/publish_release.sh --dry-run

# Publish (creates git tag + GitHub release)
bash dev/publish_release.sh
```

The script:
- Reads version from [`pyproject.toml`](../pyproject.toml)
- Validates version format (`MAJOR.MINOR.PATCH` or `MAJOR.MINOR.PATCHaN`)
- Requires being on `main` or `master` branch
- Creates a git tag and GitHub release with generated notes

> [!NOTE]
> The script requires you to change the version in `pyproject.toml` manually before running.
> Otherwise, no release nor tag is created.

## Building Documentation

```bash
# Install docs dependencies
poetry install --with docs

# Without poetry
cd docs && pip install -r requirements.txt

# Build HTML
cd docs && make html

# Clean
cd docs && make clean
```

Sphinx sources are in [`docs/source/`](../docs/source/).

## CI/CD

| Workflow | File | Trigger |
|---|---|---|
| Automated Tests | `.github/workflows/run_test.yaml` | Push to `main`; PR to `main`, `develop`, `fix/*`, `hotfix/*`, `release/*` |
| PyPI Publish | `.github/workflows/release_pypi.yaml` | Tag push (`refs/tags/*`) |
| Docs Deploy | `.github/workflows/documentation.yaml` | Push to `main` |

Tests run on Ubuntu, macOS, and Windows with Python 3.10 and 3.12.
On PRs, only Ubuntu + Python 3.10.