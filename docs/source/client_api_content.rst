Getting Started with the API Client
------------------------------------

First, import the |ApiClass| class and create an instance:

.. code-block:: python

    from datamint import Api

    api = Api()  # Uses API key from environment or config

The |ApiClass| class provides access to different endpoint handlers:

+------------------------+--------------------------------------------------+
| Property               | Purpose                                          |
+------------------------+--------------------------------------------------+
| ``api.resources``      | Upload, download, and manage data files          |
+------------------------+--------------------------------------------------+
| ``api.annotations``    | Create and manage annotations                    |
+------------------------+--------------------------------------------------+
| ``api.projects``       | Organize resources into projects                 |
+------------------------+--------------------------------------------------+
| ``api.models``         | Register and manage ML models                    |
+------------------------+--------------------------------------------------+
| ``api.deploy``         | Deploy models                                      |
+------------------------+--------------------------------------------------+
| ``api.inference``      | Run inference jobs                                 |
+------------------------+--------------------------------------------------+


Most day-to-day workflows can stay object-based. Endpoint handlers return
entity objects such as :py:class:`~datamint.entities.resource.Resource`,
:py:class:`~datamint.entities.project.Project`, and
:py:class:`~datamint.entities.annotations.annotation.Annotation`, and those entities expose
convenience methods for you to use.

Working with Resources
----------------------

Upload resource files
+++++++++++++++++++++

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
+++++++++++++++++++++++++

You can see the list of all uploaded resources by calling :py:meth:`api.resources.get_list() <datamint.api.endpoints.resources_api.ResourcesApi.get_list>`:

.. code-block:: python

    # Get resources with different filters
    inbox_resources = api.resources.get_list(status="inbox")
    dicom_resources = api.resources.get_list(mimetype="application/dicom")
    ct_resources = api.resources.get_list(channel="CT scans")

    for resource in ct_resources:
        print(resource.filename, resource.status)

Upload with options
+++++++++++++++++++

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
++++++++++++++++++

To download a resource, use :py:meth:`api.resources.download_resource_file() <datamint.api.endpoints.resources_api.ResourcesApi.download_resource_file>`:

.. code-block:: python

    # Get a resource
    resources = api.resources.get_list(status="inbox", mimetype="application/dicom")
    resource = resources[0]

    # Download as bytes through the entity helper
    bytes_obj = resource.fetch_file_data(auto_convert=False)

    # Auto-convert to the appropriate object (e.g., pydicom.Dataset)
    dicom_obj = resource.fetch_file_data(auto_convert=True)

    # Save directly to file
    resource.fetch_file_data(save_path="path/to/dicomfile.dcm")

With ``auto_convert=True``, the function uses the resource mimetype to automatically convert to the appropriate object type (``pydicom.Dataset`` for DICOM, etc.).

Publishing resources
++++++++++++++++++++

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

Deleting resources
++++++++++++++++++

To delete a resource:

.. code-block:: python

    resource = api.resources.get_list(filename="temp_file.dcm")[0]
    api.resources.delete(resource)

    # Delete multiple resources at once
    api.resources.bulk_delete(resources_to_delete)

Ranking unlabeled resources
+++++++++++++++++++++++++++

When deciding which unlabeled resources to send for annotation next, use
:py:meth:`api.resources.rank_resources() <datamint.api.endpoints.resources_api.ResourcesApi.rank_resources>`
to order them by any scoring function you provide:

.. code-block:: python

    unlabeled = api.resources.get_not_annotated(limit=200)

    ranked = api.resources.rank_resources(unlabeled, my_score_fn, top_k=20)
    for resource, score in ranked:
        print(resource.filename, score)

``rank_resources`` sorts highest score first by default (``descending=True``)
and skips any resource for which ``my_score_fn`` returns ``None``. Pass
``top_k`` to keep only the highest-ranked resources.

A common scoring function is model uncertainty -- see
:doc:`command_line_tools` and :mod:`datamint.utils.uncertainty` for how to
compute it.

Working with Annotations
------------------------

Inspect annotations from a resource
+++++++++++++++++++++++++++++++++++

Every :py:class:`~datamint.entities.resource.Resource` can fetch its own annotations:

.. code-block:: python

    resource = api.resources.get_list(project_name="Liver Review")[0]
    annotations = resource.fetch_annotations()

    for annotation in annotations:
        print(annotation.name, annotation.annotation_type)

Upload segmentations
++++++++++++++++++++

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
+++++++++++++++++++++++++

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
++++++++++++++++++++

Use :py:meth:`api.annotations.upload_volume_segmentation() <datamint.api.endpoints.annotations_api.AnnotationsApi.upload_volume_segmentation>` for NIfTI masks and other 3D segmentations:

.. code-block:: python

    volume_resource = api.resources.get_list(filename="volume.nii.gz")[0]

    api.annotations.upload_volume_segmentation(
        volume_resource,
        "path/to/segmentation.nii.gz",
        {1: "liver", 2: "tumor"},
    )

Upload geometry annotations (bounding boxes, lines)
++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: python

    from datamint.entities.annotations import BoxAnnotation, LineAnnotation, CoordinateSystem

    # Upload a bounding box
    api.annotations.upload_segmentations(
        resource,
        "path/to/box.json",
        name="tumor_box",
        annotation_type="box",
        coordinate_system=CoordinateSystem.PIXEL,
    )

Upload classification annotations
+++++++++++++++++++++++++++++++++

.. code-block:: python

    # Upload image classification labels
    api.annotations.upload_segmentations(
        resource,
        labels=["normal", "pathology"],
        name="diagnosis",
        annotation_type="category",
    )

Inspect annotation entities
+++++++++++++++++++++++++++

Annotation entities can fetch their own files and lazily resolve the source resource:

.. code-block:: python

    resource = api.resources.get_list(project_name="Liver Review")[0]
    annotation = resource.fetch_annotations(annotation_type="segmentation")[0]

    mask = annotation.fetch_file_data(use_cache=True)
    source_resource = annotation.resource

    print(annotation.name, source_resource.filename)

Measuring inter-annotator agreement
++++++++++++++++++++++++++++++++++

When a worklist assigns 2+ annotators to the same resources, use
:py:func:`~datamint.utils.annotation_agreement.compute_agreement` to quantify
how well they agree, and flag resources that need adjudication:

.. code-block:: python

    from datamint.utils.annotation_agreement import compute_agreement

    # Fetch annotations for a worklist, filtered to a single annotation type
    annotations = api.annotations.get_list(
        worklist_id=worklist.id,
        annotation_type="segmentation",
    )

    result = compute_agreement(annotations, threshold=0.7)

    print(result.overall)            # summary agreement score
    print(result.per_resource_mean)   # mean score per (resource_id, identifier)
    print(result.flagged)             # resources below the threshold

The metric is picked automatically based on the annotation type: Dice for
segmentations, IoU for bounding boxes, and Cohen's/Fleiss' kappa for
category/label annotations. Pass ``metric="dice"`` (or ``"iou"``,
``"cohen_kappa"``, ``"fleiss_kappa"``) to override the automatic choice.

With 3+ annotators, Fleiss' kappa requires a consistent count of raters per
item (not the same rater identities every time, so a pool of 5 annotators
rotating in groups of 3 per resource works fine). If rater counts vary across
items, the most common count is used for ``overall`` and items with a
different count are excluded from it, though they still appear in
``per_pair``/``per_resource_mean`` (raw pairwise agreement, useful for
flagging) marked with ``used_in_overall=False``. This means a resource can
show up as low-agreement in the table even when ``overall`` looks high.

.. automodule:: datamint.utils.annotation_agreement
   :members:
   :undoc-members:
   :show-inheritance:

Working with Projects
---------------------

Create and manage projects
++++++++++++++++++++++++++

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
++++++++++++++++++++++

The :py:class:`~datamint.entities.project.Project` entity provides shortcuts for common project workflows:

.. code-block:: python

    project = api.projects.get_by_name("My Project")

    # Cache all resource files locally for faster follow-up access
    project.cache_resources()

    resource = project.fetch_resources()[0]
    project.set_work_status(resource, "annotated")

    # Pin the metrics that matter most for this project (replaces the full list)
    project.set_pinned_metrics(["val/accuracy", "val/f1"])

    specs = project.get_annotations_specs()
    print([spec.identifier for spec in specs])

Project-scoped dataset splits
+++++++++++++++++++++++++++++

The project split endpoints return
:py:class:`~datamint.entities.project_resource_split.ProjectResourceSplit` records, which contain:

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
      - Audit metadata present when an assignment has been created.
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

    # Note: assign_splits(resources, split_name, project) — project is the third argument
    api.projects.assign_splits(train_resources, "train", project)
    api.projects.assign_splits(val_resources, "val", project)

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

Working with Channels
---------------------

A channel is just a grouping over resources, set via ``upload_channel`` at
upload time (see ``datamint upload --channel``).

.. code-block:: python

    # List channels (optionally scoped to a project)
    channels = api.resources.list_channels(project_name="MyProject")

    for channel in channels:
        print(channel.channel_name, len(channel.get_resource_ids()))

See also the tutorial notebooks: `upload_data.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/upload_data.ipynb>`_

Importing External Dataset Formats
-----------------------------------

If you already have a dataset labeled in a common format, ``datamint.importers``
(see :doc:`datamint.importers` for the full reference) saves you from
hand-rolling the upload-images-then-loop-over-annotations glue code: each
importer parses the on-disk format and uploads images plus box annotations to
a project in one call.

.. list-table::
    :header-rows: 1

    * - Importer
      - Format
      - Constructor
    * - :py:class:`~datamint.importers.coco.COCOImporter`
      - COCO JSON (``images``/``annotations``/``categories``)
      - ``COCOImporter(annotations_file, images_dir=None)``
    * - :py:class:`~datamint.importers.pascal_voc.PascalVOCImporter`
      - Pascal VOC XML (one ``.xml`` per image, ``<bndbox>`` elements)
      - ``PascalVOCImporter(annotations_dir, images_dir)``
    * - :py:class:`~datamint.importers.yolo.YOLOImporter`
      - YOLO ``.txt`` labels (normalized ``class x_center y_center width height``)
      - ``YOLOImporter(images_dir, labels_dir, class_names=None, data_yaml=None)``

Only bounding boxes are imported. If a dataset contains polygon/segmentation
annotations (or, for YOLO, OBB/keypoint label lines), ``parse()`` counts them
and ``import_to_project()`` logs a warning naming how many were skipped,
rather than failing or silently dropping them.

Every importer follows the same two-step shape: :py:meth:`~datamint.importers.coco.COCOImporter.parse`
reads and validates the dataset with no network calls (useful to preview
image/box counts and class names before uploading anything), and
:py:meth:`~datamint.importers.coco.COCOImporter.import_to_project` reuses that
parsed result to upload the images and their box annotations. ``api`` is
optional and defaults to a new :py:class:`~datamint.api.client.Api` instance
if you don't already have one:

.. code-block:: python

    from datamint import COCOImporter

    importer = COCOImporter("dataset/train/_annotations.coco.json")

    # No network calls yet -- inspect what would be uploaded
    preview = importer.parse()
    print(preview.num_images, preview.num_boxes, preview.class_names)

    # Uploads images + box annotations, reusing the parsed result above
    result = importer.import_to_project("My Project", tags=["coco-import"])
    print(result.n_images_uploaded, result.n_boxes_uploaded, result.errors)

See also the tutorial notebook: `05_import_dataset.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/03_datasets/05_import_dataset.ipynb>`_

Working with Models
--------------------

``api.models`` is a thin facade over Datamint's MLflow-backed model registry:
it wraps MLflow's ``RegisteredModel``/``ModelVersion`` objects in
:py:class:`~datamint.api.endpoints.model_types.Model` /
:py:class:`~datamint.api.endpoints.model_types.ModelVersion`, so you can
register, list, and inspect models without knowing MLflow's object model.

Register and list models
+++++++++++++++++++++++++

.. code-block:: python

    # Create a model (or fetch it if it already exists, the default behavior)
    model = api.models.create("my-model", description="Segmentation model")

    # Look up a model by name; returns None if it doesn't exist
    model = api.models.get_by_name("my-model")

    # List every registered model
    all_models = api.models.get_list()

    # Only models with a deployed image
    deployed_models = api.models.get_list(only_deployed=True)

Models are also created automatically when you pass ``--ai-model`` to
:doc:`command_line_tools` (``datamint upload``) with a name that doesn't
exist yet.

Log a model manually
+++++++++++++++++++++

Models trained through a Datamint :mod:`~datamint.lightning.trainers` are
logged automatically, annotation specs included. If you trained a model
yourself (outside a trainer) and want to register it, ``api.models.log_model()``
opens its own MLflow run and does the registration for you:

.. code-block:: python

    model = api.models.log_model(
        my_trained_model,
        project="my-project",
        model_name="my-model",
    )

To also attach annotation specs (what the model predicts, which
segmentation labels, box classes, or categories) without building
``AnnotationSpec`` objects by hand, pass the dataset you trained on. Specs are
derived from its actual labels, dispatching on ``my_trained_model.task_type``:

.. code-block:: python

    from datamint.dataset import ImageDataset

    dataset = ImageDataset(project="my-project", return_boxes=True)

    model = api.models.log_model(
        my_trained_model,
        project="my-project",
        model_name="my-model",
        dataset=dataset,
    )

Passing ``annotation_specs`` explicitly always takes precedence over anything
derived from ``dataset``. Omitting ``dataset`` altogether logs the model with
no annotation specs, same as calling
:func:`~datamint.mlflow.flavors.datamint_flavor.log_model` directly.

Inspect versions and metrics
++++++++++++++++++++++++++++

Each :py:class:`~datamint.api.endpoints.model_types.Model` can list its
:py:class:`~datamint.api.endpoints.model_types.ModelVersion` objects, and each
version exposes what it was trained for and how it performed:

.. code-block:: python

    model = api.models.get_by_name("my-model")

    versions = model.get_versions()
    latest = model.get_latest_version()          # highest version number
    champion = model.get_latest_version(alias="champion")

    print(latest.get_task_type())                 # e.g. "segmentation"
    print(latest.get_supported_modes())            # e.g. ["auto", "interactive"]
    print(latest.get_metrics())                    # e.g. {"val/dice": 0.87}

``get_metrics()`` returns ``{}`` for versions with no training run behind
them (for example, a model registered externally rather than trained through
a Datamint :mod:`~datamint.lightning.trainers`), rather than raising.
``Model.get_supported_modes()``/``get_metrics()`` are shortcuts that delegate
to the latest version when you don't need a specific one.

``latest.load_model()`` loads the model itself, ready for local inference.

Find which projects a model belongs to
++++++++++++++++++++++++++++++++++++++

.. code-block:: python

    model = api.models.get_by_name("my-model")
    projects = model.get_projects()          # list[Project]

Model registry (MLflow) operations
++++++++++++++++++++++++++++++++++

``api.models`` wraps the underlying MLflow model registry client directly, so
these calls map one-to-one onto MLflow's own registry API:

+---------------------------------------------+-------------------------------------------+
| Method                                      | MLflow equivalent                         |
+=============================================+===========================================+
| ``api.models.create(...)``                  | ``MlflowClient.create_registered_model``  |
+---------------------------------------------+-------------------------------------------+
| ``api.models.get_by_name(...)``             | ``MlflowClient.get_registered_model``     |
+---------------------------------------------+-------------------------------------------+
| ``api.models.get_list(...)``                | ``MlflowClient.search_registered_models`` |
+---------------------------------------------+-------------------------------------------+
| ``api.models.delete_model_version(...)``    | ``MlflowClient.delete_model_version``     |
+---------------------------------------------+-------------------------------------------+
| ``api.models.delete_registered_model(...)`` | ``MlflowClient.delete_registered_model``  |
+---------------------------------------------+-------------------------------------------+

.. code-block:: python

    # Delete a single version
    api.models.delete_model_version("my-model", version=1)

    # Delete a registered model and all of its remaining versions
    api.models.delete_registered_model("my-model")

Clone a model to another project
++++++++++++++++++++++++++++++++

Models live in the MLflow registry of whichever project is active (see
:func:`datamint.mlflow.set_project`), so using a model trained in one project
against another normally means manually reloading and re-logging it.
``api.models.clone_model()`` does that for you:

.. code-block:: python

    from datamint.mlflow import set_project

    set_project("Project A")  # clone_model() resolves the source here

    cloned = api.models.clone_model(
        "my-model",
        target_project="Project B",
        version=3,                        # or alias="champion"
        target_model_name="my-model-v2",  # optional, defaults to the source name
    )

Only models logged with the ``datamint`` MLflow flavor are supported, since
that's what carries the task type, supported modes, and annotation specs the
clone copies over. If the source model's class depends on custom code (not an
installed package), pass ``code_paths`` the same way you would to
``log_model()`` -- otherwise the clone registers successfully but fails to
load later, since its class gets pickled by reference to code the new
artifact never bundled:

.. code-block:: python

    api.models.clone_model(
        "my-model",
        target_project="Project B",
        version=3,
        code_paths=["my_adapter.py"],
    )

The project that was active before the call is always restored afterward,
even if cloning fails partway through.

Deploy a registered model
+++++++++++++++++++++++++

Use ``api.deploy.start()`` to deploy a model:

.. code-block:: python

    # Deploy a registered model
    deploy_job = api.deploy.start(
        model_name="my-model",
        model_alias="latest",
    )
    print(deploy_job.status)

    # Wait for deployment to complete
    deploy_job = deploy_job.wait()
    print("Deployment complete:", deploy_job.status)

    # Check whether a model has a deployed image
    model.is_deployed()

Working with Users
------------------

User management operations:

.. code-block:: python

    # List all users
    users = api.users.get_all()

    # Get user by email (email serves as the entity ID)
    user = api.users.get_by_email("user@example.com")
