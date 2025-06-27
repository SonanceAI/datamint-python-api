
# Datamint python API

![Build Status](https://github.com/SonanceAI/datamint-python-api/actions/workflows/run_test.yaml/badge.svg)

See the full documentation at https://sonanceai.github.io/datamint-python-api/

## Installation

Datamint requires Python 3.10+.
You can install/update Datamint and its dependencies using pip

```bash
pip install -U datamint
```

We recommend that you install Datamint in a dedicated virtual environment, to avoid conflicting with your system packages.
Create the enviroment once with `python3 -m venv datamint-env` and then activate it whenever you need it with:
- `source datamint-env/bin/activate` (Linux/MAC)
- `datamint-env\Scripts\activate.bat` (Windows CMD)
- `datamint-env\Scripts\Activate.ps1` (Windows PowerShell)


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

### Method 3: APIHandler constructor

Specify API key in the |APIHandlerClass| constructor:

```python
from datamint import APIHandler
api = APIHandler(api_key='my_api_key')
```

## Tutorials


You can find example notebooks in the `notebooks` folder:

- [Uploading your resources](notebooks/upload_data.ipynb)
- [Uploading model segmentations](notebooks/upload_model_segmentations.ipynb)

and example scripts in [examples](examples) folder:

- [Running an experiment for classification](examples/experiment_traintest_classifier.py)
- [Running an experiment for segmentation](examples/experiment_traintest_segmentation.py)

## Full documentation

See all functionalities in the full documentation at https://sonanceai.github.io/datamint-python-api/
