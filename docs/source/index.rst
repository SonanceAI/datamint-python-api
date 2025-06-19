.. DatamintAPI documentation master file


.. image:: ../images/logo.png
   :height: 250
   :align: center
   :alt: Datamint

Documentation
=======================================
**Version:** |release|

From inception to completion, Datamint is your reliable partner.
It assists from the very first day when you make your data available to your team, right up to the moment you're set to launch your model.

Datamint
--------

- `Homepage <https://www.datamint.io>`_
- `Datamint Platform <https://app.datamint.io/>`_
- `Github <https://github.com/SonanceAI/datamint-python-api>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   command_line_tools
   client_api
   pytorch_integration


.. toctree::
   :maxdepth: 1
   :caption: Python Modules Reference

   datamint.apihandler
   datamint.dataset

Super Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install git+https://github.com/SonanceAI/datamint-python-api

Configure your API access:

.. code-block:: bash

   datamint-config

Start using the API:

.. code-block:: python

   from datamintapi import APIHandler, Dataset
   
   # Initialize API handler
   api = APIHandler()
   
   # Load a dataset
   dataset = Dataset(project_name="your-project")
   
   # Access your data
   for sample in dataset:
       image = sample['image']
       segmentations = sample['segmentations']

Key Features
------------

- **Dataset Management**: Download, upload, and manage medical imaging datasets
- **Annotation Tools**: Create, upload, and manage annotations (segmentations, labels, measurements)
- **Experiment Tracking**: Integrated MLflow support for experiment management
- **PyTorch Lightning Integration**: Streamlined ML workflows with Lightning DataModules and callbacks
- **DICOM Support**: Native handling of DICOM files with anonymization capabilities
- **Multi-format Support**: PNG, JPEG, NIfTI, and other medical imaging formats


Examples Gallery
----------------

Medical Image Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datamint.lightning import DatamintDataModule
   from datamint.mlflow.lightning.callbacks import MLFlowModelCheckpoint
   import lightning as L

   # Setup data module
   datamodule = DatamintDataModule(
       project_name="liver-segmentation",
       batch_size=16,
       train_split=0.8
   )

   # Train with MLflow tracking
   trainer = L.Trainer(
       callbacks=[MLFlowModelCheckpoint(
           monitor="val_dice",
           register_model_name="liver-model"
       )]
   )

Batch Resource Upload
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Upload DICOM files with anonymization
   datamint-upload \
       --path /data/dicoms \
       --recursive \
       --anonymize \
       --channel "training-data"

Annotation Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Upload segmentation masks
   api.upload_segmentations(
       resource_id="resource-123",
       file_path="liver_mask.nii.gz",
       name="liver_segmentation"
   )

   # Add categorical labels
   api.add_image_category_annotation(
       resource_id="resource-123",
       identifier="pathology",
       value="cirrhosis"
   )

Community & Support
-------------------
`GitHub Issues <https://github.com/SonanceAI/datamint-python-api/issues>`_

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
