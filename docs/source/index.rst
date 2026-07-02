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
   datamint_vs_raw_pytorch
   ssl_troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Python Modules Reference

   datamint.apihandler
   datamint.api.base_classes
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

The SDK is built around a few core concepts that make data ingestion, annotation, training, and deployment work together smoothly.

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Resources
      :link: datamint.entities.resource.Resource
      :link-type: any

      **Manage source data.**

      Upload medical images, videos, and other files. Organize resources into **channels** and **projects**, then tag and annotate them.

   .. grid-item-card:: Annotations
      :link: datamint.entities.annotations.annotation.Annotation
      :link-type: any

      **Capture labels and geometry.**

      Add segmentations, bounding boxes, classifications, and other geometry to resources, with support for both 2D images and 3D volumes.

   .. grid-item-card:: Projects
      :link: datamint.entities.project.Project
      :link-type: any

      **Group data for workflows.**

      Collect resources for annotation and ML training. Projects support **split assignments** (train/val/test) to keep experiments reproducible.

   .. grid-item-card:: Datasets
      :link: datamint.dataset.base.DatamintBaseDataset
      :link-type: any

      **Train with PyTorch-ready data.**

      Use dataset classes that load data from Datamint projects and automatically handle DICOM, NIfTI, image, and video formats.

   .. grid-item-card:: Trainers
      :link: datamint.lightning.trainers.BaseTrainer
      :link-type: any

      **Accelerate common training loops.**

      Rely on high-level trainers to streamline dataset setup, model configuration, MLflow logging, and checkpointing.

   .. grid-item-card:: Models
      :link: datamint.mlflow.flavors.model.DatamintModel
      :link-type: any

      **Package models for deployment.**

      Register ML models for inference on the Datamint platform, including segmentation, classification, and other custom use cases.

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
   deploy_job = api.deploy.start(
       model_name="liver-segmentation-model",
       model_alias="latest",
   )
   print(deploy_job.status)

Community & Support
-------------------

`GitHub Issues <https://github.com/SonanceAI/datamint-python-api/issues>`_

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
