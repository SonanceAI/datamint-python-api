Getting Started with Datamint Python API
=========================================

This guide will help you set up and start using the Datamint Python API for your medical imaging projects.

Prerequisites
=============

- **Python 3.10 or later** (earlier versions are not supported)
- **pip** or **conda** for package management
- A **Datamint account** with an API key (get one at `app.datamint.io <https://app.datamint.io/>`_)

Installation
============

Datamint requires Python 3.10+.
Install/update Datamint and its dependencies using pip

.. code-block:: bash

    pip install -U datamint

We recommend that you install Datamint in a dedicated virtualenv, to avoid conflicting with your system packages.
You can do this by running:

.. code-block:: bash

    python3 -m venv datamint-env
    source datamint-env/bin/activate  # In Windows, run datamint-env\Scripts\activate.bat
    pip install -U datamint

Verify your installation
------------------------

.. code-block:: bash

    python -c "import datamint; print(datamint.__version__)"
    datamint-config --help

.. include:: setup_api_key.rst

Troubleshooting
---------------

.. dropdown:: ImportError: No module named 'datamint'
    :icon: wrench

    Make sure you activated your virtual environment before running your script:

    .. code-block:: bash

        source datamint-env/bin/activate  # Linux/macOS
        datamint-env\Scripts\activate     # Windows

    Or install globally (not recommended):

    .. code-block:: bash

        pip install --user -U datamint

.. dropdown:: API authentication errors
    :icon: wrench

    Verify your API key is set correctly:

    .. code-block:: bash

        datamint-config --show-api-key

    Or check the environment variable:

    .. code-block:: bash

        echo $DATAMINT_API_KEY

.. dropdown:: DICOM upload fails with pydicom errors
    :icon: wrench

    Make sure you have the latest ``pydicom`` and ``medimgkit`` packages:

    .. code-block:: bash

        pip install -U pydicom medimgkit

.. dropdown:: MLflow configuration issues
    :icon: wrench

    The MLflow module auto-configures itself on first import. If you encounter issues:

    .. code-block:: python

        from datamint.mlflow import ensure_mlflow_configured
        ensure_mlflow_configured()

Next Steps
----------

Now that you have the basics set up, explore these advanced topics:

- Master the command-line interface: :ref:`command_line_tools`
- Check out our Python API documentation: :ref:`client_python_api`
- Our PyTorch, Lightning and MLflow integration: :ref:`pytorch_integration`
- Use the built-in Trainer API and custom model integration patterns: :ref:`trainer_api`
- Browse tutorial notebooks: :doc:`tutorials`
