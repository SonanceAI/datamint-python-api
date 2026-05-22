Entities
========

The ``datamint.entities`` module provides the core data structures that represent
various objects within the Datamint ecosystem. These entities are built using `Pydantic <https://pydantic-docs.helpmanual.io/>`_ models, ensuring robust data validation,
type safety, and seamless serialization/deserialization when interacting with the Datamint API.

In most user code, prefer working with entity instances instead of passing raw IDs
between endpoint calls. :py:class:`~datamint.entities.project.Project`,
:py:class:`~datamint.entities.resource.Resource`, and
:py:class:`~datamint.entities.annotations.annotation.Annotation` can usually be passed directly back into
API methods, and they expose convenience helpers for common workflows.

Entity-first Workflows
----------------------

Project objects
+++++++++++++++

.. code-block:: python

   from datamint import Api

   api = Api()
   project = api.projects.get_by_name("Liver Review")

   resources = project.fetch_resources()
   project.cache_resources(progress_bar=False)

   if resources:
      project.set_work_status(resources[0], "annotated")

   specs = project.get_annotations_specs()
   print([spec.identifier for spec in specs])

Resource objects
++++++++++++++++

.. code-block:: python

   resource = api.resources.get_list(project_name="Liver Review")[0]

   image = resource.fetch_file_data(auto_convert=True, use_cache=True)
   annotations = resource.fetch_annotations(annotation_type="segmentation")

   print(resource.filename, len(annotations))

Annotation objects
++++++++++++++++++

.. code-block:: python

   annotation = resource.fetch_annotations()[0]

   annotation_data = annotation.fetch_file_data(use_cache=True)
   source_resource = annotation.resource

   print(annotation.name, source_resource.filename)

Entity Reference
----------------

Base Classes
++++++++++++

.. automodule:: datamint.entities.base_entity
   :members: BaseEntity, BaseEntityModel
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.entities.cache_manager
   :members: CacheManager
   :undoc-members:
   :show-inheritance:

Resource Entities
+++++++++++++++++

.. automodule:: datamint.entities.resource
   :members: Resource, LocalResource
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.entities.resources.dicom_resource
   :members: DICOMResource
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.entities.resources.image_resource
   :members: ImageResource
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.entities.resources.nifti_resource
   :members: NiftiResource
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.entities.resources.video_resource
   :members: VideoResource
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.entities.resources.volume_resource
   :members: VolumeResource
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.entities.sliced_resource
   :members: SlicedResource
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.entities.sliced_video_resource
   :members: SlicedVideoResource
   :undoc-members:
   :show-inheritance:

Project Entity
++++++++++++++

.. automodule:: datamint.entities.project
   :members:
   :undoc-members:
   :show-inheritance:

Channel Entity
++++++++++++++

.. automodule:: datamint.entities.channel
   :members: Channel, ChannelResourceData
   :undoc-members:
   :show-inheritance:

User Entity
+++++++++++

.. automodule:: datamint.entities.user
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Info Entity
+++++++++++++++++++

.. automodule:: datamint.entities.datasetinfo
   :members: DatasetInfo
   :undoc-members:
   :show-inheritance:

Split Entity
++++++++++++

.. automodule:: datamint.entities.project_resource_split
   :members:
   :undoc-members:
   :show-inheritance:

Job Entities
++++++++++++

.. automodule:: datamint.entities.deployjob
   :members: DeployJob
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.entities.inferencejob
   :members: InferenceJob
   :undoc-members:
   :show-inheritance:

Annotations Subpackage
----------------------

The ``datamint.entities.annotations`` subpackage contains all annotation-related
entity classes. See :doc:`datamint.entities.annotations` for detailed documentation.

.. automodule:: datamint.entities.annotations
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
