# Datamint Python API

![Build Status](https://github.com/SonanceAI/datamint-python-api/actions/workflows/run_test.yaml/badge.svg)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python SDK for interacting with the Datamint platform, providing seamless integration for medical imaging workflows, dataset management, and machine learning experiments.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Setup](#-setup-api-key)
- [Documentation](#-documentation)
- [Command Line Tools](#️-command-line-tools)
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

`pip install -U datamint`

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
   pip install datamint
   ```

</details>

## ⚙ Setup API key

To use the Datamint API, you need to setup your API key (ask your administrator if you don't have one). Use one of the following methods to setup your API key:

### Method 1: Command-line tool (recommended)

Run ``datamint-config`` in the terminal and follow the instructions. See [command_line_tools](https://sonanceai.github.io/datamint-python-api/command_line_tools.html#configuring-the-datamint-settings) for more details.

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
| [🚀 Getting Started](https://sonanceai.github.io/datamint-python-api/getting_started.html) | Step-by-step setup and basic usage |
| [📖 API Reference](https://sonanceai.github.io/datamint-python-api/client_api.html) | Complete API documentation |
| [🔥 PyTorch Integration](https://sonanceai.github.io/datamint-python-api/pytorch_integration.html) | ML workflow integration |
| [💡 Examples](examples/) | Practical usage examples |

## 🛠️ Command Line Tools

Full documentation at [command_line_tools](https://sonanceai.github.io/datamint-python-api/command_line_tools.html).

### Upload Resources

**Upload DICOM files with anonymization:**
```bash
datamint-upload /path/to/dicoms --recursive --channel "training-data" --publish --tag "my_data_tag"
```
It anonymizes by default.

### Configuration Management

```bash
# Interactive setup
datamint-config

# Set API key
datamint-config --api-key "your-key"
```

## 🔒 SSL Certificate Troubleshooting

If you encounter SSL certificate verification errors like:
```
SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

### Quick Fix

**1. Upgrade certifi:**
```bash
pip install --upgrade certifi
```

**2. Set environment variables:**
```bash
export SSL_CERT_FILE=$(python -m certifi)
export REQUESTS_CA_BUNDLE=$(python -m certifi)
```

**3. Run your script:**
```bash
python your_script.py
```

### Alternative Solutions

**Option 1: Use Custom CA Bundle**
```python
from datamint import Api

api = Api(verify_ssl="/path/to/your/ca-bundle.crt")
```

**Option 2: Disable SSL Verification (Development Only)**
```python
from datamint import Api

# ⚠️ WARNING: Only use in development with self-signed certificates
api = Api(verify_ssl=False)
```

## 🆘 Support

[Full Documentation](https://datamint-python-api.readthedocs.io/)  
[GitHub Issues](https://github.com/SonanceAI/datamint-python-api/issues)

