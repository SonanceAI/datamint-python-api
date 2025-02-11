Installation
===================================

Datamint requires Python 3.10+.
Install Datamint and its dependencies using pip

.. code-block:: bash

    pip install git+https://github.com/SonanceAI/datamint-python-api

Soon we will be releasing the package on PyPi.
We recommend that you install Datamint in a dedicated virtualenv, to avoid conflicting with your system packages.
You can do this by running:

.. code-block:: bash

    python3 -m venv datamint-env
    source datamint-env/bin/activate # In Windows, run datamint-env\Scripts\activate.bat
    pip install git+https://github.com/SonanceAI/datamint-python-api



Next Steps
------------
- Setup your API key: :ref:`setup_api_key`
- Check out our command line tools: :ref:`command_line_tools`
- Check out our Python API documentation: :ref:`client_python_api`
- Our Pytorch integration: :ref:`pytorch_integration`