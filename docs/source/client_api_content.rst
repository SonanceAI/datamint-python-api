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
- ``api.models`` - For managing registered models
- ``api.annotationsets`` - For working with annotation set configurations
- ``api.deploy`` - For deploying models to the Datamint platform
- ``api.inference`` - For running and managing inference jobs

Most day-to-day workflows can stay object-based. Endpoint handlers return
entity objects such as :py:class:`~datamint.entities.Resource`,
:py:class:`~datamint.entities.Project`, and
:py:class:`~datamint.entities.Annotation`, and those entities expose
convenience methods for you to use.

Working with Resources
----------------------

Upload resource files
++++++++++++++++++++++++++++++++

Use :py:meth:`api.resources.upload_resource() <datamint.api.endpoints.resources_api.ResourcesApi.upload_resource>` to upload any resource type, such as DICOMs, videos, and image files:

.. code-block:: python

    # Upload a single file
    api.resources.upload_resource("/path/to/dicom.dcm")

    # Upload multiple files at once
    api.resources.upload_resources([
        "/path/to/dicom.dcm",
        "/path/to/video.mp4",
    ])

List and filter resources
++++++++++++++++++++++++++++++++

You can see the list of all uploaded resources by calling :py:meth:`api.resources.get_list() <datamint.api.endpoints.resources_api.ResourcesApi.get_list>`:

.. code-block:: python

    # Get resources with different filters
    inbox_resources = api.resources.get_list(status="inbox")
    dicom_resources = api.resources.get_list(mimetype="application/dicom")
    ct_resources = api.resources.get_list(channel="CT scans")

    for resource in ct_resources:
        print(resource.filename, resource.status)

Upload with options
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

You can customize the upload with various parameters:

.. code-block:: python

    # Upload with channel organization
    api.resources.upload_resource(
        "/path/to/dicom.dcm",
        channel="CT scans",
    )

    # Upload with anonymization and labels
    api.resources.upload_resource(
        "/path/to/dicom.dcm",
        anonymize=True,
        tags=["baseline", "ct"],
    )

    # Upload and publish directly to a project
    project = api.projects.get_by_name("Liver Review")
    api.resources.upload_resource(
        "/path/to/dicom.dcm",
        publish_to=project,
    )

Download resources
------------------

To download a resource, use :py:meth:`api.resources.download_resource_file() <datamint.api.endpoints.resources_api.ResourcesApi.download_resource_file>`:

.. code-block:: python

    # Get a resource
    resources = api.resources.get_list(status="inbox", mimetype="application/dicom")
    resource = resources[0]

    # Download as bytes through the entity helper
    bytes_obj = resource.fetch_file_data(auto_convert=False)

    # Auto-convert to the appropriate object (for example pydicom.Dataset)
    dicom_obj = resource.fetch_file_data(auto_convert=True)

    # Save directly to file
    resource.fetch_file_data(save_path="path/to/dicomfile.dcm")

With ``auto_convert=True``, the function uses the resource mimetype to automatically convert to the appropriate object type (``pydicom.Dataset`` for DICOM, etc.).

Publishing resources
---------------------

To publish a resource, use :py:meth:`api.resources.publish_resources() <datamint.api.endpoints.resources_api.ResourcesApi.publish_resources>`:

.. code-block:: python

    resources = api.resources.get_list(status="inbox")
    resource = resources[0]  # assuming there is at least one resource in the inbox

    # Change status from 'inbox' to 'published'
    api.resources.publish_resources(resource)

    # Add the published resource to a project
    project = api.projects.get_by_name("Liver Review")
    api.projects.add_resources(resource, project)

If you want the resource to land directly in a project, prefer
``upload_resource(..., publish_to=project)`` during upload.

Working with Annotations
------------------------

Inspect annotations from a resource
++++++++++++++++++++++++++++++++

Every :py:class:`~datamint.entities.Resource` can fetch its own annotations:

.. code-block:: python

    resource = api.resources.get_list(project_name="Liver Review")[0]
    annotations = resource.fetch_annotations()

    for annotation in annotations:
        print(annotation.name, annotation.annotation_type)

Upload segmentations
++++++++++++++++++++++++++++++++

To upload a segmentation, use :py:meth:`api.annotations.upload_segmentations() <datamint.api.endpoints.annotations_api.AnnotationsApi.upload_segmentations>`:

.. code-block:: python

    resource = api.resources.get_list(filename="dicom.dcm")[0]

    # Upload segmentation
    api.annotations.upload_segmentations(
        resource,
        "path/to/segmentation.png",
        name="SegmentationName",
    )

Multi-class segmentations
++++++++++++++++++++++++++++++++

If your segmentation has multiple classes, you can pass a dictionary mapping pixel values to class names:

.. code-block:: python

    class_names = {
        # Background (0) is automatic, don't specify it
        1: "tumor",
        2: "vessel",
    }

    api.annotations.upload_segmentations(
        resource,
        "path/to/segmentation.png",
        name=class_names,
    )

Volume segmentations
++++++++++++++++++++++++++++++++

Use :py:meth:`api.annotations.upload_volume_segmentation() <datamint.api.endpoints.annotations_api.AnnotationsApi.upload_volume_segmentation>` for NIfTI masks and other 3D segmentations:

.. code-block:: python

    volume_resource = api.resources.get_list(filename="volume.nii.gz")[0]

    api.annotations.upload_volume_segmentation(
        volume_resource,
        "path/to/segmentation.nii.gz",
        {1: "liver", 2: "tumor"},
    )

Inspect annotation entities
++++++++++++++++++++++++++++++++

Annotation entities can fetch their own files and lazily resolve the source resource:

.. code-block:: python

    resource = api.resources.get_list(project_name="Liver Review")[0]
    annotation = resource.fetch_annotations(annotation_type="segmentation")[0]

    mask = annotation.fetch_file_data(use_cache=True)
    source_resource = annotation.resource

    print(annotation.name, source_resource.filename)

Working with Projects
---------------------

Create and manage projects
++++++++++++++++++++++++++++++++

.. code-block:: python

    # Create a new project
    project = api.projects.create(
        name="My Project",
        description="Project description",
    )

    # Add existing resources to it
    resources = api.resources.get_list(channel="CT scans")
    api.projects.add_resources(resources, project)

    # Work with project resources through the entity
    for resource in project.fetch_resources():
        print(resource.filename)

Project helper methods
++++++++++++++++++++++++++++++++

The :py:class:`~datamint.entities.Project` entity provides shortcuts for common project workflows:

.. code-block:: python

    project = api.projects.get_by_name("My Project")

    # Cache all resource files locally for faster follow-up access
    project.cache_resources()

    resource = project.fetch_resources()[0]
    project.set_work_status(resource, "annotated")

    specs = project.get_annotations_specs()
    print([spec.identifier for spec in specs])

Project-scoped dataset splits
++++++++++++++++++++++++++++++++

The project split endpoints return
:py:class:`~datamint.entities.ProjectResourceSplit` records, which contain:

.. list-table::
     :header-rows: 1

     * - Field
         - Description
     * - ``split_name``
         - Logical split name such as ``train``, ``val``, or ``test``.
     * - ``project_id``
         - Project that owns the assignment.
     * - ``resource_id``
         - Resource assigned within that project.
     * - ``created_at`` / ``created_by``
         - Audit metadata for assignment creation.
     * - ``deleted_at`` / ``deleted_by``
         - Audit metadata present when an assignment has been deleted.

Use :py:meth:`api.projects.assign_splits() <datamint.api.endpoints.projects_api.ProjectsApi.assign_splits>`
to write assignments, :py:meth:`api.projects.get_splits() <datamint.api.endpoints.projects_api.ProjectsApi.get_splits>`
to list them, and :py:meth:`api.projects.get_resource_split() <datamint.api.endpoints.projects_api.ProjectsApi.get_resource_split>`
to inspect one resource within a project:

.. code-block:: python

    from datamint import Api

    api = Api()
    project = api.projects.get_by_name("FracAtlas")
    resources = list(project.fetch_resources())

    train_resources = resources[:100]
    val_resources = resources[100:120]

    api.projects.assign_splits(project, train_resources, "train")
    api.projects.assign_splits(project, val_resources, "val")

    assignments = api.projects.get_splits(project)
    train_assignments = api.projects.get_splits(project, split_name="train")
    first_resource_assignment = api.projects.get_resource_split(project, resources[0])

For project-backed datasets, :py:meth:`~datamint.dataset.base.DatamintBaseDataset.split`
now prefers project-scoped assignments automatically when you do not pass
ratio kwargs:

.. code-block:: python

    from datamint.dataset import ImageDataset

    dataset = ImageDataset(project=project, include_unannotated=True)

    parts = dataset.split()
    snapshot = parts["train"].split_as_of_timestamp

    # Reuse the exact assignment snapshot later.
    replayed_parts = dataset.split(as_of_timestamp=snapshot)

Each returned subset records ``split_name``, ``split_source``, and
``split_as_of_timestamp`` for reproducibility. Local ratio splits remain
available with calls such as ``dataset.split(train=0.8, val=0.2, seed=42)``.
Legacy ``split:*`` tag-based splitting is still supported for backwards
compatibility, but it is deprecated in favor of project-scoped splits.

Working with Channels
---------------------

Organize resources with channels
++++++++++++++++++++++++++++++++

.. code-block:: python

    # List all channels
    channels = api.channels.get_list()

    # Create a new channel
    api.channels.create(name="CT Scans", description="CT scan images")

See also the tutorial notebooks: `upload_data.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/upload_data.ipynb>`_

Working with Models & Deployment
---------------------------------

Register and deploy models via ``api.models`` and ``api.deploy``:

.. code-block:: python

    # List all registered models
    models = api.models.get_list()

    # Deploy a registered model
    deploy_job = api.deploy.deploy(model_name="my-model", version="1.0")
    print(deploy_job.status)

Running inference
+++++++++++++++++

Use ``api.inference`` to trigger inference jobs against deployed models:

.. code-block:: python

    resource = api.resources.get_list(project_name="Liver Review")[0]
    job = api.inference.run(model_name="my-model", resource=resource)
    print(job.status)

See the tutorial notebooks: `deploy_model_demo.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/deploy_model_demo.ipynb>`_
and `external_model_deployment_tutorial.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/external_model_deployment_tutorial.ipynb>`_

