Getting Started with the API Client
------------------------------------

First, import the |ApiClass| class and create an instance:

.. code-block:: python

    from datamint import Api
    api = Api()  # Uses API key from environment or config

The |ApiClass| class provides access to different endpoint handlers:

- ``api.resources`` - For uploading, downloading, and managing resources
- ``api.annotations`` - For creating and managing annotations/segmentations  
- ``api.projects`` - For creating and managing projects
- ``api.channels`` - For organizing resources into channels
- ``api.users`` - For user management operations

Working with Resources
----------------------

Upload resource files
++++++++++++++++++++++++++++++++

Use :py:meth:`api.resources.upload_resource() <datamint.api.endpoints.resources_api.ResourcesApi.upload_resource>` to upload any resource type, such as DICOMs, videos, and image files:

.. code-block:: python

    # Upload a single file
    resource_id = api.resources.upload_resource("/path/to/dicom.dcm")

    # Upload multiple files at once
    resource_ids = api.resources.upload_resources(["/path/to/dicom.dcm", 
                                                   "/path/to/video.mp4"])

List and filter resources
++++++++++++++++++++++++++++++++

You can see the list of all uploaded resources by calling :py:meth:`api.resources.get_list() <datamint.api.endpoints.resources_api.ResourcesApi.get_list>`:

.. code-block:: python

    # Get resources with different filters
    resources = api.resources.get_list(status='inbox')  # status: 'inbox', 'published', 'archived'
    resources = api.resources.get_list(mimetype='application/dicom')  # filter by mimetype
    resources = api.resources.get_list(channel='CT scans')  # filter by channel
    
    for resource in resources:
        print(f"Resource {resource.id}: {resource.filename}")

Upload with options
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can customize the upload with various parameters:

.. code-block:: python

    # Upload with channel organization
    resource_id = api.resources.upload_resource("/path/to/dicom.dcm",
                                                 channel='CT scans')

    # Upload with anonymization and labels
    resource_id = api.resources.upload_resource("/path/to/dicom.dcm",
                                                 anonymize=True,
                                                 tags=['label1', 'label2'])

    # Upload and publish directly to a project
    resource_id = api.resources.upload_resource("/path/to/dicom.dcm",
                                                 publish=True,
                                                 publish_to='ProjectName')

Download resources
------------------

To download a resource, use :py:meth:`api.resources.download_resource_file() <datamint.api.endpoints.resources_api.ResourcesApi.download_resource_file>`:

.. code-block:: python

    # Get a resource
    resources = api.resources.get_list(status='inbox', mimetype='application/dicom')
    resource = resources[0]

    # Download as bytes
    bytes_obj = api.resources.download_resource_file(resource.id, auto_convert=False)

    # Auto-convert to appropriate object (e.g., pydicom.Dataset for DICOM files)
    dicom_obj = api.resources.download_resource_file(resource.id, auto_convert=True)
        
    # Save directly to file
    api.resources.download_resource_file(resource.id, save_path='path/to/dicomfile.dcm')

With ``auto_convert=True``, the function uses the resource mimetype to automatically convert to the appropriate object type (``pydicom.Dataset`` for DICOM, etc.).

Publishing resources
---------------------

To publish a resource, use :py:meth:`api.resources.publish_resources() <datamint.api.endpoints.resources_api.ResourcesApi.publish_resources>`:

.. code-block:: python

    resources = api.resources.get_list(status='inbox')
    resource = resources[0]  # assuming there is at least one resource in the inbox

    # Change status from 'inbox' to 'published'
    api.resources.publish_resources(resource.id)

    # Publish to a specific project
    api.resources.publish_resources(resource.id, project_name='ProjectName')

Working with Annotations
------------------------

Upload segmentations
++++++++++++++++++++++++++++++++

To upload a segmentation, use :py:meth:`api.annotations.upload_segmentations() <datamint.api.endpoints.annotations_api.AnnotationsApi.upload_segmentations>`:

.. code-block:: python
    
    # Upload a resource first (or use an existing resource_id)
    resource_id = api.resources.upload_resources("/path/to/dicom.dcm")
    
    # Upload segmentation
    api.annotations.upload_segmentations(resource_id, 
                                        'path/to/segmentation.nii.gz',  # NIfTI or PNG file
                                        name='SegmentationName')

Multi-class segmentations
++++++++++++++++++++++++++++++++

If your segmentation has multiple classes, you can pass a dictionary mapping pixel values to class names:

.. code-block:: python

    class_names = {
        # Background (0) is automatic, don't specify it
        1: "tumor",
        2: "metal",
    }

    api.annotations.upload_segmentations(resource_id, 
                                        'path/to/segmentation.nii.gz',
                                        name=class_names)

Working with Projects
---------------------

Create and manage projects
++++++++++++++++++++++++++++++++

.. code-block:: python

    # Create a new project
    project_id = api.projects.create(
        name='My Project',
        description='Project description',
        resources_ids=[resource_id1, resource_id2]  # optional
    )

    # Get project details
    project = api.projects.get_by_id(project_id)
    
    # List all projects
    projects = api.projects.get_list()
    
    # Get resources in a project
    project_resources = api.projects.get_project_resources(project_id)

Use :py:meth:`api.projects.create() <datamint.api.endpoints.projects_api.ProjectsApi.create>` to create projects, ``api.projects.get_by_id()`` to retrieve them, and :py:meth:`api.projects.get_project_resources() <datamint.api.endpoints.projects_api.ProjectsApi.get_project_resources>` to get associated resources.

Working with Channels  
---------------------

Organize resources with channels
++++++++++++++++++++++++++++++++

.. code-block:: python

    # List all channels
    channels = api.channels.get_list()
    
    # Create a new channel
    channel_id = api.channels.create(name='CT Scans', description='CT scan images')

See also the tutorial notebooks: `upload_data.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/upload_data.ipynb>`_
