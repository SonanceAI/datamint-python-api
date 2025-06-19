Getting Started with Datamint Python API
=========================================

This guide will help you set up and start using the Datamint Python API for your medical imaging projects.

Installation
------------

System Requirements
~~~~~~~~~~~~~~~~~~~

- Python 3.10 or higher
- Operating System: Windows, macOS, or Linux

Install
~~~~~~~~~~~~~~~~~

The easiest way to install the Datamint Python API:

.. code-block:: bash

    pip install git+https://github.com/SonanceAI/datamint-python-api

Soon we will be releasing the package on PyPi.
We recommend that you install Datamint in a dedicated virtualenv, to avoid conflicting with your system packages.
You can do this by running:

.. code-block:: bash

    python3 -m venv datamint-env
    source datamint-env/bin/activate # In Windows, run datamint-env\Scripts\activate.bat
    pip install git+https://github.com/SonanceAI/datamint-python-api


.. include:: setup_api_key.rst

Next Steps
------------
Now that you have the basics set up, explore these advanced topics:

- Master the command-line interface: :ref:`command_line_tools`
- Check out our Python API documentation: :ref:`client_python_api`
- Our Pytorch, Lightning and MLflow integration: :ref:`pytorch_integration`