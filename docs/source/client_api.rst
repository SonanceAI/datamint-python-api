.. _client_python_api:

Client API
==========

This section describes how to use the :py:class:`APIHandler <datamintapi.api_handler.APIHandler>` class in Python.

Setup API key
-------------
There are three options to specify the API key:

1. **Recommended:** Run ``datamint-config`` and follow the instructions. See :ref:`configuring_datamint_settings` for more details.
2. Specify API key as an environment variable:

.. tabs:: 

    .. code-tab:: bash

        export DATAMINT_API_KEY="my_api_key"
        python my_script.py

    .. code-tab:: python

        os.environ["DATAMINT_API_KEY"] = "my_api_key"
    
3. Specify API key in the :py:class:`APIHandler <datamintapi.api_handler.APIHandler>` constructor:

.. code-block:: python

   from datamintapi import APIHandler

   api_handler = APIHandler(api_key='my_api_key')

Upload DICOMs and other resources
----------------------------------

First, import the :py:class:`APIHandler <datamintapi.api_handler.APIHandler>` class and create an instance: ``api_handler = APIHandler(...)``.
This class is responsible for interacting with the Datamint server.

Upload resource files
++++++++++++++++++++++++++++++++

Use the :py:meth:`upload_resources() <datamintapi.api_handler.APIHandler.upload_resources>` method to upload any resource type, such as DICOMs, videos, image files:

.. code-block:: python

    # Upload a single file
    resource_id = api_handler.upload_resources("/path/to/dicom.dcm")

    # Upload multiple files at once
    resoures_ids = api_handler.upload_resources(["/path/to/dicom.dcm", 
                                                 "/path/to/video.mp4"]
                                                )

You can see the list of all uploaded resources by calling the :py:meth:`get_resources() <datamintapi.api_handler.APIHandler.get_resources>` method:

.. code-block:: python

    resources = api_handler.get_resources(status='inbox') # status can be any of {'inbox', 'published', 'archived'}
    # Alternatively, you can use apihandler.get_resources_by_ids(resources_ids)
    for res in resources:
        print(res)

Group up resources using channels
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For a better organization of resources, you can group them into channels:

.. code-block:: python

    # Uploads a resource and creates a new channel named 'CT scans':
    resource_id = api_handler.upload_resources("/path/to/dicom.dcm",
                                               channel='CT scans'
                                               )

    # This uploads a new resource to the same channel:
    resource_id = api_handler.upload_resources("/path/to/dicom2.dcm",
                                               channel='CT scans'
                                               )                              
    
    # Get all resources from channel 'CT scans':
    resources = api_handler.get_resources(channel='CT scans')
    

Upload, anonymize and add a label
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To anonymize and add a label to a DICOM file, use the parameters `anonymize`
and `labels` of :py:meth:`upload_resources() <datamintapi.api_handler.APIHandler.upload_upload_resourcesdicom>`:

.. code-block:: python

    dicom_id = api_handler.upload_resources(files_path='/path/to/video_data.mp4',
                                        anonymize=True,
                                        labels=['pneumonia'])



Create a batch and upload DICOMs
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A batch is a collection of DICOM files.
To create a batch, use the :py:meth:`create_batch() <datamintapi.api_handler.APIHandler.create_batch>` method.

.. code-block:: python

    batch_id = api_handler.create_batch(description='CT scans',
                                        size=3)

With the batch created, use the batch_id to upload DICOM files to the batch:

.. code-block:: python

    dicom_id = api_handler.upload_dicom(batch_id=batch_id, 
                                        file_path=file_path)

To create a batch and upload dicoms in a single call, use :py:meth:`create_batch_with_dicoms() <datamintapi.api_handler.APIHandler.create_batch_with_dicoms>`,

.. code-block:: python

    batch_id, dicoms_ids = api_handler.create_batch_with_dicoms(description='CT scans',
                                                                file_path='/path/to/dicom_files/',
                                                                mung_filename='all',
                                                                )

, which outputs the batch_id and the list of dicom_ids that were uploaded.
The `mung_filename='all'` parameters in this example converts the files names into 'path_to_dicom_files/1.dcm', 'path_to_dicom_files/2.dcm', etc.

Download resources
------------------

To download a resource, use the :py:meth:`download_resources() <datamintapi.api_handler.APIHandler.download_resources>` method:

.. code-block:: python

    resources = api_handler.get_resources(status='inbox', mimetype='application/dicom')
    resource_id = resources[0]['id']

    dicom_obj = api_handler.download_resources(resource_id)
    # dicom_obj is dicom object (pydicom.Dataset)

The function above uses the resource mimetype to automatically convert to a proper object type (`pydicom.Dataset`, in this case.)
If you do not this, but the bytes itself, use the ``auto_convert=False`` parameter.
Additionally, if you want to save the file to disk, use the ``save_path`` parameter:

.. code-block:: python

    dicom_bytes = api_handler.download_resources(resource_id, 
                                                 auto_convert=False,
                                                 save_path='path/to/save/dicom.dcm')
    # dicom_bytes is a bytes object


Publishing a resource
---------------------

To publish a resource, use the :py:meth:`publish_resource() <datamintapi.api_handler.APIHandler.publish_resource>` method:

.. code-block:: python

    resources = api_handler.get_resources(status='inbox')
    resource_id = resources[0]['id'] # assuming there is at least one resource in the inbox

    api_handler.publish_resource(resource_id)

You can also publish resources while uploading them:

.. code-block:: python

    resource_id = api_handler.upload_resources(files_path='/path/to/video_data.mp4',
                                               publish=True
                                               )

Upload segmentation
-------------------

To upload a segmentation, use the :py:meth:`upload_segmentation() <datamintapi.api_handler.APIHandler.upload_segmentation>` method:

.. code-block:: python
    
    resource_id = api_handler.upload_resources("/path/to/dicom1.dcm") # or use an existing resource_id
    api_handler.upload_segmentation(resource_id, 'path/to/segmentation.nifti', 'SegmentationName')


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

Here is a complete example that inherits :py:class:`datamintapi.dataset.DatamintDataset`:

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