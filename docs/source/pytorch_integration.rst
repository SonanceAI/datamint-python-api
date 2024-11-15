.. _pytorch_integration:

Pytorch integration
===================

Before continuing, you may want to check the :ref:`setup_api_key` section to easily set up your API key, if you haven't done so yet.

Dataset
-------

Datamint provides a custom PyTorch dataset class that can be used to load data from the server in a PyTorch-friendly way.
To use it, import the :py:class:`~datamintapi.dataset.DatamintDataset` class and create an instance of it, passing the necessary parameters.

.. code-block:: python

    from datamintapi import Dataset

    dataset = Dataset('../data',
                      project_name='MyProjectName', # Must exists in the server
                      # return_frame_by_frame=True, # Optional, if you want each item to be a frame instead of a video/3d-image
                     )

and then use it in your PyTorch code as usual.

Here is a complete example that inherits :py:class:`~datamintapi.dataset.DatamintDataset`:

.. code-block:: python

    import datamintapi
    import torch
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader


    class XrayFractureDataset(datamintapi.Dataset):
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

    import datamintapi
    import torch
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader


    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Create an instance of the datamintapi.Dataset
    dataset = datamintapi.Dataset(root='data',
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