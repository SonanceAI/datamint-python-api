.. DatamintAPI documentation master file

.. image:: ../images/logo.png
   :height: 250
   :align: center
   :alt: Datamint

Documentation
=======================================
**Version:** |release|

A comprehensive Python SDK for interacting with the Datamint platform, providing seamless integration for medical imaging workflows, dataset management, and machine learning experiments.

From inception to completion, Datamint is your reliable partner.
It assists from the very first day when you make your data available to your team, right up to the moment you're set to launch your model.

Datamint
--------

- `Homepage <https://www.datamint.io>`_
- `Datamint Platform <https://app.datamint.io/>`_
- `GitHub <https://github.com/SonanceAI/datamint-python-api>`_

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   command_line_tools
   client_api
   pytorch_integration
   trainer_api
   tutorials

.. toctree::
   :maxdepth: 1
   :caption: Python Modules Reference

   datamint.apihandler
   datamint.dataset
   datamint.entities
   datamint.lightning_api
   datamint.mlflow_api
   datamint.exceptions

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install datamint

Configure your API access:

.. code-block:: bash

   datamint-config

Start using the API:

.. code-block:: python

   from datamint import Api

   # Initialize API handler
   api = Api()
   all_projects = api.projects.get_all()

   # Upload a resource
   api.resources.upload_resource("/path/to/image.dcm")

   # Load a dataset for training
   from datamint.dataset import ImageDataset
   dataset = ImageDataset(project="my-project")

Architecture Overview
---------------------

The Datamint Python API is organized into several key modules:

+------------------------+--------------------------------------------------+-----------------------------+
| Module                 | Purpose                                          | Key Classes                 |
+========================+==================================================+=============================+
| ``datamint.api``       | HTTP client and endpoint handlers for the API    | ``Api``, ``ResourcesApi``   |
|                        |                                                  | ``ProjectsApi``, etc.       |
+------------------------+--------------------------------------------------+-----------------------------+
| ``datamint.entities``  | Pydantic data models representing platform       | ``Resource``, ``Project``   |
|                        | objects                                          | ``Annotation``, etc.        |
+------------------------+--------------------------------------------------+-----------------------------+
| ``datamint.dataset``   | PyTorch dataset classes for medical imaging      | ``ImageDataset``,           |
|                        |                                                  | ``VolumeDataset``, etc.     |
+------------------------+--------------------------------------------------+-----------------------------+
| ``datamint.lightning`` | PyTorch Lightning integration for training       | ``DatamintDataModule``,     |
|                        | workflows                                        | ``UNetPPTrainer``, etc.     |
+------------------------+--------------------------------------------------+-----------------------------+
| ``datamint.mlflow``    | MLflow integration for experiment tracking and   | ``DatamintMLflowDataset``,  |
|                        | model registration                               | ``DatamintModel``, etc.     |
+------------------------+--------------------------------------------------+-----------------------------+

Key Concepts
------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Resources
      :link: :py:class:`~datamint.entities.resource.Resource`

      Uploaded medical images, videos, and other data files. Resources can be organized into **channels** and **projects**, tagged, and annotated.

   .. grid-item-card:: Annotations
      :link: :py:class:`~datamint.entities.annotations.annotation.Annotation`

      Segmentations, bounding boxes, classifications, and geometry attached to resources. Supports 2D image and 3D volume segmentations.

   .. grid-item-card:: Projects
      :link: :py:class:`~datamint.entities.project.Project`

      Collections of resources used for annotation workflows and ML training. Projects support **split assignments** (train/val/test) for reproducible experiments.

   .. grid-item-card:: Datasets
      :link: :py:class:`~datamint.dataset.base.DatamintBaseDataset`

      PyTorch-compatible dataset classes that load data from Datamint projects. Automatically handle DICOM, NIfTI, image, and video formats.

   .. grid-item-card:: Trainers
      :link: :py:class:`~datamint.lightning.trainers.BaseTrainer`

      High-level trainers that automate dataset creation, model configuration, MLflow logging, and checkpointing for common tasks.

   .. grid-item-card:: Models
      :link: :py:class:`~datamint.mlflow.flavors.model.DatamintModel`

      Registered ML models that can be deployed on the Datamint platform for inference. Supports segmentation, classification, and custom models.

Common Workflows
----------------

Uploading Data
~~~~~~~~~~~~~~

.. code-block:: python

   from datamint import Api

   api = Api()

   # Upload a single file
   resource = api.resources.upload_resource("/path/to/image.dcm")

   # Upload with options
   api.resources.upload_resource(
       "/path/to/image.dcm",
       channel="CT Scans",
       tags=["baseline", "ct"],
       anonymize=True,
   )

   # Upload multiple files
   api.resources.upload_resources(["/path/to/a.dcm", "/path/to/b.dcm"])

Creating a Training Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamint import Api
   from datamint.dataset import ImageDataset

   api = Api()

   # Create project
   project = api.projects.create(
       name="Liver Segmentation",
       description="CT liver segmentation dataset",
   )

   # Add resources
   resources = api.resources.get_list(channel="CT Scans")
   api.projects.add_resources(resources, project)

   # Load dataset
   dataset = ImageDataset(project="Liver Segmentation")

Training a Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamint.lightning import UNetPPTrainer

   trainer = UNetPPTrainer(
       project="Liver Segmentation",
       image_size=256,
       batch_size=16,
       max_epochs=50,
       accelerator="gpu",
   )

   results = trainer.fit()
   print(results["test_results"])

Deploying a Model
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamint import Api

   api = Api()

   # Deploy a registered model
   deploy_job = api.deploy.deploy(
       model_name="liver-segmentation",
       version="1.0.0",
   )
   print(deploy_job.status)

   # Run inference
   resource = api.resources.get_list(filename="patient001.dcm")[0]
   job = api.inference.run(model_name="liver-segmentation", resource=resource)

Community & Support
-------------------

`GitHub Issues <https://github.com/SonanceAI/datamint-python-api/issues>`_

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
