.. _pytorch_integration:


PyTorch & Lightning Integration
===============================

The Datamint Python API provides seamless integration with PyTorch and PyTorch Lightning, enabling efficient machine learning workflows for medical imaging tasks.

Overview
--------

Key integration features:

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
   from datamint import Dataset
   
   # Load dataset. This is a PyTorch-compatible dataset that can be used directly.
   dataset = Dataset(
       project_name="liver-segmentation",
       return_annotations=True,
       return_frame_by_frame=True,
       include_unannotated=False
   )
   
   # Create PyTorch DataLoader
   dataloader = DataLoader(
       dataset,
       batch_size=16,
       shuffle=True,
       num_workers=4,
       collate_fn=dataset.get_collate_fn()
   )
   
   # Training loop
   for batch in dataloader:
       images = batch['image']      # Shape: [B, C, H, W]
       masks = batch['segmentation'] # Shape: [B, H, W]
       metadata = batch['metainfo']  # List of dicts
       # (...)

Dataset Transforms
~~~~~~~~~~~~~~~~~~

Apply transforms for data augmentation and preprocessing:

.. code-block:: python

    import datamint
    import torch
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader


    class XrayFractureDataset(datamint.Dataset):
        def __getitem__(self, idx):
            image, dicom_metainfo, metainfo = super().__getitem__(idx)

            # Get all relevant information from the dicom_metainfo object
            patient_sex = dicom_metainfo.PatientSex

            # Get all relevant information from the metainfo object
            has_fracture = 'fracture' in metainfo['labels']
            has_fracture = torch.tensor(has_fracture, dtype=torch.int32)

            return image, patient_sex, has_fracture


    # Create an instance of your custom dataset
    dataset = XrayFractureDataset(root='data',
                                  dataset_name='YOUR_DATASET_NAME',
                                  version='latest',
                                  api_key='my_api_key',
                                  transform=ToTensor())

    # Create a DataLoader to handle batching and shuffling of the dataset
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True)

    for images, patients_sex, labels in dataloader:
        images = images.to(device)
        # labels will already be a tensor of shape (batch_size,) containing 0s and 1s

        # (...) do something with the batch

Alternative code, if you want to load all the data and metadata:

.. code-block:: python

    import datamint
    import torch
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader


    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Create an instance of the datamint.Dataset
    dataset = datamint.Dataset(root='data',
                                dataset_name='TestCTdataset',
                                version='latest',
                                api_key='my_api_key',
                                transform=ToTensor()
                                )

    # This function tells the dataloader how to group the items in a batch
    def collate_fn(batch):
        images = [item[0] for item in batch]
        dicom_metainfo = [item[1] for item in batch]
        metainfo = [item[2] for item in batch]

        return torch.stack(images), dicom_metainfo, metainfo


    # Create a DataLoader to handle batching and shuffling of the dataset
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            collate_fn=collate_fn,
                            shuffle=True)

    for images, dicom_metainfo, metainfo in dataloader:
        images = images.to(device)
        metainfo = metainfo

        # (... do something with the batch)