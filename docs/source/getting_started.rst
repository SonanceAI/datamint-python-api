Quick Start
=========================================

This guide will help you set up and start using the Datamint Python API for your medical imaging projects.

Installation
============

Datamint requires Python 3.10+ and a `Datamint account <https://app.datamint.io/>`_ with an API key.

.. code-block:: bash

    pip install -U datamint

We recommend a dedicated virtualenv to avoid conflicting with your system packages:

.. code-block:: bash

    python3 -m venv datamint-env
    source datamint-env/bin/activate  # Windows: datamint-env\Scripts\activate.bat
    pip install -U datamint

Verify your installation
------------------------

.. code-block:: bash

    python -c "import datamint; print(datamint.__version__)"

.. include:: setup_api_key.rst

Scaffold your first project
===========================

``datamint-init`` generates a ready-to-run set of numbered scripts tailored to your task
(detection, segmentation, or classification):

.. code-block:: bash

    datamint-init

It asks for a project name and task type, then writes six scripts into a new directory
(upload data, explore, build a dataset, train, evaluate, and deploy), so you can follow
them in order without writing boilerplate.

Your first API call
===================

Once installed and configured, verify everything works end-to-end:

.. code-block:: python

    from datamint import Api

    api = Api()
    for project in api.projects.get_all():
        print(project.name)

.. tip::

    Want to see full end-to-end examples? Browse our :doc:`tutorial notebooks <tutorials>` —
    they cover real datasets, training workflows, and deployment from scratch.

Next Steps
----------

- Master the command-line interface: :ref:`command_line_tools`
- Check out our Python API documentation: :ref:`client_python_api`
- Our PyTorch, Lightning and MLflow integration: :ref:`pytorch_integration`
- Use the built-in Trainer API and custom model integration patterns: :ref:`trainer_api`

Troubleshooting
---------------

.. dropdown:: ImportError: No module named 'datamint'
    :icon: tools

    Make sure you activated your virtual environment before running your script:

    .. code-block:: bash

        source datamint-env/bin/activate  # Linux/macOS
        datamint-env\Scripts\activate     # Windows

    Or install globally (not recommended):

    .. code-block:: bash

        pip install --user -U datamint

.. dropdown:: API authentication errors
    :icon: tools

    Verify your API key is set correctly:

    .. code-block:: bash

        datamint-config
