# Datamint python API

See the full documentation at https://sonanceai.github.io/datamint-python-api/

Installation
------------

Datamint requires Python 3.8+. You can install Datamint and its
dependencies using pip

```bash
pip install git+https://github.com/SonanceAI/datamint-python-api
```

Soon we will be releasing the package on PyPi.
We recommend that you install Datamint in a dedicated virtual environment, to avoid conflicting with your system packages.
Create the enviroment once with `python3 -m venv datamint-env` and then activate it whenever you need it with:
- `source datamint-env/bin/activate` (Linux/MAC)
- `datamint-env\Scripts\activate.bat` (Windows CMD)
- `datamint-env\Scripts\Activate.ps1` (Windows PowerShell)


Setup API key
-------------

There are three options to specify the API key:

1. **Recommended**: Run `datamint-config` command-line and follow the instructions
2.  Specify the API key as an environment variable:  

**bash**
```bash
export DATAMINT_API_KEY="my_api_key"
python my_script.py
```

**python**
```python
os.environ["DATAMINT_API_KEY"] = "my_api_key"
```

3.  Specify API key in the [APIHandler](datamintapi/api_handler.py) constructor:

```python
from datamintapi import APIHandler
api_handler = APIHandler(api_key='my_api_key')
```

Tutorials
---------

You can find example notebooks in the `notebooks` folder:

- [Uploading your resources](notebooks/upload_data.ipynb)
- [Uploading model segmentations](notebooks/upload_model_segmentations.ipynb)

and example scripts in [examples](examples) folder:

- [Running an experiment for classification](examples/experiment_traintest_classifier.py)
- [Running an experiment for segmentation](examples/experiment_traintest_segmentation.py)

Full documentation
---------------------

See all functionalities in the full documentation at https://sonanceai.github.io/datamint-python-api/
