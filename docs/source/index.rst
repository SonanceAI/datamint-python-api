.. DatamintAPI documentation master file

.. image:: ../images/logo_no_bg.png
   :width: 150px
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

**Difficulty levels:**

- :bdg-success:`Beginner` no ML knowledge needed
- :bdg-warning:`Intermediate` assumes SDK familiarity, introduces ML/dataset concepts
- :bdg-danger:`Advanced` full training pipelines, custom models, 3D data, multi-step workflows

For Developers
--------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Quick Start
      :link: getting_started
      :link-type: doc

      :bdg-success:`Beginner`

      Install the package, configure your API key, and make your first calls.

   .. grid-item-card:: Command-line tools
      :link: command_line_tools
      :link-type: doc

      :bdg-success:`Beginner`

      Run the unified ``datamint <command>`` CLI for uploads, config, training, and inference.

   .. grid-item-card:: Client Python API
      :link: client_api
      :link-type: doc

      :bdg-warning:`Intermediate`

      Use the ``Api`` class directly for full control over resources, projects, and annotations.

   .. grid-item-card:: PyTorch & Lightning Integration
      :link: pytorch_integration
      :link-type: doc

      :bdg-warning:`Intermediate`

      Plug Datamint datasets into PyTorch and Lightning training loops.

   .. grid-item-card:: Training your Model
      :link: trainer_api
      :link-type: doc

      :bdg-warning:`Intermediate`

      Train models with built-in one-line trainers -- no training loop to write.

   .. grid-item-card:: Tutorials
      :link: tutorials
      :link-type: doc

      Browse runnable example notebooks, from getting started through full end-to-end pipelines.

   .. grid-item-card:: Datamint vs Raw PyTorch
      :link: datamint_vs_raw_pytorch
      :link-type: doc

      :bdg-danger:`Advanced`

      Side-by-side comparison of raw PyTorch/Lightning code vs the Datamint equivalent.

   .. grid-item-card:: SSL Troubleshooting
      :link: ssl_troubleshooting
      :link-type: doc

      Fix ``SSLCertVerificationError`` issues when connecting to the API.

.. toctree::
   :maxdepth: 2
   :caption: For Developers
   :hidden:

   getting_started
   command_line_tools
   client_api
   pytorch_integration
   trainer_api
   tutorials
   datamint_vs_raw_pytorch
   ssl_troubleshooting

For Non-Developers
-------------------

Just need to upload images and segmentations, no coding? Start here.

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Upload Images & Segmentations
      :link: for_non_devs
      :link-type: doc

      :bdg-success:`Beginner`

      No coding required. Set up your computer, get an API key, and upload your files
      from a terminal, step by step.

.. toctree::
   :maxdepth: 2
   :caption: For Non-Developers
   :hidden:

   for_non_devs

Quick Start
-----------

.. code-block:: bash

   pip install datamint
   datamint config

.. code-block:: python

   from datamint import Api

   api = Api()
   all_projects = api.projects.get_all()

See the full :doc:`Quick Start guide <getting_started>` for installing in a virtual
environment, configuring your API key, and scaffolding a project with ``datamint init``.

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

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
