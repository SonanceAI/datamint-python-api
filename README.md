# Datamint python API

See the full documentation at https://sonanceai.github.io/datamint-python-api/

Installation
------------

Datamint requires Python 3.8+. You can install Datamint and its
dependencies using pip

``` {.bash}
pip install git+https://github.com/SonanceAI/datamint-python-api
```

Soon we will be releasing the package on PyPi. We strongly recommend
that you install Datamint in a dedicated virtualenv, to avoid
conflicting with your system packages.

Setup API key
-------------

There are two options to specify the API key:

1.  Specify API key as an environment variable:
```bash
export DATAMINT_API_KEY="my_api_key" python my_script.py
```

```python
os.environ["DATAMINT_API_KEY"] = "my_api_key"
```

2.  Specify API key in the
    :py`APIHandler <datamintapi.api_handler.APIHandler>`{.interpreted-text
    role="class"} constructor:

```python
from datamintapi import APIHandler
api_handler = APIHandler(api_key='my_api_key')
```

Other functionalities
---------------------

See the full documentation at https://sonanceai.github.io/datamint-python-api/