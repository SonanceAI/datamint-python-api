Upload DICOMs or other resources
----------------------------------

First, import the |APIHandlerClass| class and create an instance: ``api_handler = APIHandler(...)``.
This class is responsible for interacting with the Datamint server.

Upload resource files
++++++++++++++++++++++++++++++++

Use the :py:meth:`upload_resources() <datamint.apihandler.api_handler.APIHandler.upload_resources>` method to upload any resource type, such as DICOMs, videos, and image files:

.. code-block:: python

    # Upload a single file
    resource_id = api_handler.upload_resources("/path/to/dicom.dcm")

    # Upload multiple files at once
    resoures_ids = api_handler.upload_resources(["/path/to/dicom.dcm", 
                                                 "/path/to/video.mp4"]
                                                )

You can see the list of all uploaded resources by calling the :py:meth:`get_resources() <datamint.apihandler.api_handler.APIHandler.get_resources>` method:

.. code-block:: python

    resources = api_handler.get_resources(status='inbox') # status can be any of {'inbox', 'published', 'archived'}
    for res in resources:
        print(res)
    # Alternatively, you can use apihandler.get_resources_by_ids(resources_ids)

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

To anonymize and add labels to a DICOM file, use the parameters `anonymize`
and `labels` of :py:meth:`upload_resources() <datamint.apihandler.api_handler.APIHandler.upload_resources>`.
Adding labels is useful for searching and filtering resources in the Datamint platform later.

.. code-block:: python

    dicom_id = api_handler.upload_resources(files_path='/path/to/dicom.dcm',
                                            anonymize=True,
                                            labels=['label1', 'label2']
                                            )



Changing the uploaded filename
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

By default, the filename that is uploaded is the basename of the file. 
For instance, if you upload a file named 'path/to/dicom.dcm', the filename will be 'dicom.dcm'.
To include the path into the filename, use the `mung_filename` parameter:

.. code-block:: python

    # filename='dicom.dcm' (DEFAULT)
    resource_ids = api_handler.upload_resources(files_path='path/to/dicom.dcm',
                                                mung_filename=None,
                                                )

    # filename='path_to_dicom.dcm'
    resource_ids = api_handler.upload_resources(files_path='path/to/dicom.dcm',
                                                mung_filename='all',
                                                )

    # filename='to_dicom.dcm'
    resource_ids = api_handler.upload_resources(files_path='path/to/dicom.dcm',
                                                mung_filename=1,
                                                )



Download resources
------------------

To download a resource, use the :py:meth:`~datamint.apihandler.api_handler.APIHandler.download_resource_file` method:

.. code-block:: python

    resources = api_handler.get_resources(status='inbox', mimetype='application/dicom')
    resource_id = resources[0]['id']

    # returns the resource content in bytes:
    bytes_obj = api_handler.download_resource_file(resource_id, auto_convert=False)

    # Assuming this resource is a dicom file, it will return a pydicom.dataset.Dataset object. 
    dicom_obj = api_handler.download_resource_file(resource_id, auto_convert=True)
        
    # saves the file in the specified path.
    api_handler.download_resource_file(resource_id, save_path='path/to/dicomfile.dcm')
        
With ``auto_convert=True``, the function uses the resource mimetype to automatically convert to a proper object type (`pydicom.dataset.Dataset`, in this case.)
If you do not want this, but the bytes itself, use the ``auto_convert=False``.


Publishing resources
---------------------

To publish a resource, use :py:meth:`~datamint.apihandler.api_handler.APIHandler.publish_resources`:

.. code-block:: python

    resources = api_handler.get_resources(status='inbox')
    resource_id = resources[0]['id'] # assuming there is at least one resource in the inbox

    # Change status from 'inbox' to 'published'
    api_handler.publish_resources(resource_id)

To publish to a project, pass the project name or id as an argument:

.. code-block:: python

    api_handler.publish_resources(resource_id, project_name='ProjectName')

You can also publish resources while uploading them:

.. code-block:: python

    resource_id = api_handler.upload_resources(files_path='/path/to/video_data.mp4',
                                               publish=True,
                                               # publish_to='ProjectName' # optional
                                               )

Upload segmentation
-------------------

To upload a segmentation, use :py:meth:`upload_segmentations() <datamint.apihandler.api_handler.APIHandler.upload_segmentations>`:

.. code-block:: python
    
    resource_id = api_handler.upload_resources("/path/to/dicom1.dcm") # or use an existing resource_id
    api_handler.upload_segmentations(resource_id, 
                                    'path/to/segmentation.nii.gz', # Can be a nifti file or an png file
                                     name='SegmentationName')


If your segmentation has multiple classes, you can pass a dictionary mapping pixel values to class names.
Let's say you have a segmentation with 2 classes, where pixel value 0 is background, 1 is 'tumor', and 2 is 'metal':

.. code-block:: python

    class_names = {
        # Do not specify the background class, it is always 0 
        1: "tumor",
        2: "metal",
    }

    api_handler.upload_segmentations(resource_id, 
                                    'path/to/segmentation.nii.gz', # Can be a nifti file or an png file
                                     name=class_names
                                    )

See also the tutorial notebook on uploading data: `upload_data.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/upload_data.ipynb>`_
