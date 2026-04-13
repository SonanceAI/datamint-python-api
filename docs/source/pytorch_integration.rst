.. _pytorch_integration:


PyTorch & Lightning Integration
===============================

The Datamint Python API provides seamless integration with PyTorch and PyTorch Lightning, enabling efficient machine learning workflows for medical imaging tasks.

Overview
--------

Key integration features:

- **ImageDataset / VolumeDataset / VideoDataset**: Modular, PyTorch-compatible datasets for 2D images, 3D volumes, and video sequences
- **DatamintDataModule**: Lightning-compatible data module
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

            # 'metainfo' is a dict with file/DICOM metadata
            metainfo = item['metainfo']

            has_fracture = 'fracture' in item.get('image_labels', [])
            label = torch.tensor(has_fracture, dtype=torch.int32)

            return image, label


    # Create an instance of your custom dataset
    dataset = XrayFractureDataset(
        project='YOUR_PROJECT_NAME',
        api_key='my_api_key',
        alb_transform=A.Compose([A.Resize(224, 224)]),
    )

    # Create a DataLoader to handle batching and shuffling of the dataset
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
        # images: (batch_size, C, H, W)
        # labels: (batch_size,)
        pass  # (...) do something with the batch

Loading all data and metadata:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    from datamint.dataset import ImageDataset

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create an instance of ImageDataset
    dataset = ImageDataset(
        project='my_project_name',
        api_key='my_api_key',
    )

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        images = batch['image'].to(device)  # (batch_size, C, H, W)
        metainfo = batch['metainfo']        # list of metadata dicts

        # (... do something with the batch)