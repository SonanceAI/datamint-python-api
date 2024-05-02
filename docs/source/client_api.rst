Client API
==========

Import the APIHandler class and create an instance: `api_handler = APIHandler()`

Setup API key
-------------
.. note:: Not required if you don't plan to communicate with the server.

There are three options to specify the API key:

1. Specify API key as an environment variable:
    - **command line:** ``export DATAMINT_API_KEY="my_api_key"; python my_script.py``
    - **python:** ``os.environ["DATAMINT_API_KEY"] = "my_api_key"``
2. Specify API key in the API Handler constructor:

.. code-block:: python

   from datamintapi import APIHandler

   api_handler = APIHandler(api_key='my_api_key')

3. run ``datamint config``? (TODO?) and follow the instructions?

Upload dicoms
-------------

To upload dicoms, use the `upload_dicoms` method of the `APIHandler` class.

To upload a single dicom file:

.. code-block:: python

    dicom_id = api_handler.upload_dicoms(batch_id='abcd1234', 
                                         file_path="/path/to/dicom.dcm")
    

To upload a dicom, anonymize it and add label 'pneumonia' to it:

.. code-block:: python

    dicom_id = api_handler.upload_dicom(batch_id='abcd1234', 
                                        file_path=file_path,
                                        anonymize=True,
                                        labels=['pneumonia'])

**To upload a directory of dicoms, while creating a new batch in a single call:**

.. code-block:: python

    api_handler.create_batch_with_dicoms(description='CT scans',
                                         file_path='/path/to/dicom_files/',
                                         mung_filename='all', # This will convert files name to 'path_to_dicom_files/1.dcm', 'path_to_dicom_files/2.dcm', etc.
                                         ):



Dataset
-------

Datamint provides a custom PyTorch dataset class that can be used to load data from the server.
To use it, import the custom dataset class and create an instance: 

.. code-block:: python

    from datamintapi import Dataset

    dataset = Dataset(root='../data',
                     dataset_name='TestCTdataset',
                     version='latest',
                     api_key='my_api_key'
                     )

and then use it in your PyTorch code as usual.

Here is an complete example that inherits :py:class:`datamintapi.dataset.DatamintDataset`:

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

Alternative if you don't want to inherit from :py:class:`datamintapi.dataset.DatamintDataset`:

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