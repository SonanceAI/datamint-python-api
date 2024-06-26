# Datamint python API

See the full documentation at https://sonanceai.github.io/datamint-python-api/

Installation
------------

Datamint requires Python 3.8+. You can install Datamint and its
dependencies using pip

```bash
pip install git+https://github.com/SonanceAI/datamint-python-api
```

Soon we will be releasing the package on PyPi. We strongly recommend
that you install Datamint in a dedicated virtualenv, to avoid
conflicting with your system packages.

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

3.  Specify API key in the [APIHandler](datamintapi/api_handler.py#L95) constructor:

```python
from datamintapi import APIHandler
api_handler = APIHandler(api_key='my_api_key')
```

Other functionalities
---------------------

See the full documentation at https://sonanceai.github.io/datamint-python-api/
