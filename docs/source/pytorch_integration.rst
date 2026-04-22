.. _pytorch_integration:


PyTorch & Lightning Integration
===============================

The Datamint Python API provides seamless integration with PyTorch and PyTorch Lightning, enabling efficient machine learning workflows for medical imaging tasks.

Overview
--------

Key integration features:

- **ImageDataset / VolumeDataset / VideoDataset**: Modular, PyTorch-compatible datasets for 2D images, 3D volumes, and video sequences
- **DatamintDataModule**: Lightning-compatible data module
- **Trainer API**: Task-focused trainers such as ``UNetPPTrainer`` and ``SemanticSegmentation2DTrainer``
- **MLFlowModelCheckpoint**: Advanced model checkpointing with MLflow integration
- **Automatic Experiment Tracking**: Seamless logging and model registration
- **Medical Image Optimizations**: Specialized handling for medical data formats

PyTorch Dataset Integration
---------------------------

Basic PyTorch Usage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from datamint.dataset import ImageDataset, VolumeDataset

   # 2D images (X-rays, single-frame DICOM, PNG, JPEG, …)
   dataset = ImageDataset(
       project="liver-classification",
       include_unannotated=False,
   )

   # 3D volumes (NIfTI, DICOM series, …)
   # dataset = VolumeDataset(project="ct-liver-segmentation")

   # Create a standard PyTorch DataLoader
   dataloader = DataLoader(
       dataset,
       batch_size=16,
       shuffle=True,
       num_workers=4,
   )

   # Training loop – each batch is a dict
   for batch in dataloader:
       images = batch['image']           # Shape: [B, C, H, W]
       segmentations = batch['segmentations']  # Shape varies by mode
       metadata = batch['metainfo']      # List of dicts
       # (...)

Dataset Transforms
~~~~~~~~~~~~~~~~~~

Apply transforms for data augmentation and preprocessing:

.. code-block:: python

    import albumentations as A
    import torch
    from torch.utils.data import DataLoader
    from datamint.dataset import ImageDataset


    class XrayFractureDataset(ImageDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)

            # 'image' is a tensor of shape (C, H, W)
            image = item['image']

            has_fracture = 'fracture' in item.get('image_labels', [])
            label = torch.tensor(has_fracture, dtype=torch.int32)

            return image, label


    # Create an instance of your custom dataset
    dataset = XrayFractureDataset(
        project='MY_PROJECT_NAME',
        alb_transform=A.Compose([A.Resize(224, 224)]),
    )

    # Create a DataLoader to handle batching and shuffling of the dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
        # images: (batch_size, C, 224, 224)
        # labels: (batch_size,)
        pass  # (...) do something with the batch

Loading all data and metadata:

.. code-block:: python

    from datamint.dataset import ImageDataset

    # Create an instance of ImageDataset
    dataset = ImageDataset(
        project='MY_PROJECT_NAME'
    )
    # Create a DataLoader
    dataloader = dataset.get_dataloader(batch_size=4, shuffle=False) # Same parameters as PyTorch DataLoader

    for batch in dataloader:
        images = batch['image']  # shape: (batch_size, C, H, W)
        segmentations = batch['segmentations']
        image_labels = batch['image_labels']
        image_categories = batch['image_categories']

        # (... do something with the batch)

Split Reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :py:meth:`~datamint.dataset.base.DatamintBaseDataset.split`
for using split assignments.
The resolved split datasets store the effective historical
snapshot timestamp in ``split_as_of_timestamp``, which you can pass back into
future training runs to reuse the exact same assignment state.

.. code-block:: python

   from datamint.dataset import ImageDataset
   from datamint.lightning import DatamintDataModule

   dataset = ImageDataset(project="my-project", include_unannotated=True)

   parts = dataset.split()
   snapshot = parts["train"].split_as_of_timestamp

   datamodule = DatamintDataModule(
       dataset,
       split=True,
       split_as_of_timestamp=snapshot,
   )

``DatamintDataModule`` and the built-in trainers propagate the resolved
``split_source`` and ``split_as_of_timestamp`` values into MLflow lineage, so
you can trace which project split snapshot was used during training and replay
it later.

Trainer API
-----------

Use the Trainer API when you want Datamint to build the dataset, datamodule,
default model, MLflow logger, and checkpoint callbacks for you.

.. code-block:: python

   from datamint.lightning import UNetPPTrainer

   trainer = UNetPPTrainer(
       project="BUSI_Segmentation",
       image_size=256,
       batch_size=16,
       max_epochs=20,
       accelerator="auto",
   )
   results = trainer.fit()

The trainer layer is also the recommended way to integrate an external model
architecture while still reusing Datamint's dataset handling and MLflow
workflow.

See :ref:`trainer_api` for more details.