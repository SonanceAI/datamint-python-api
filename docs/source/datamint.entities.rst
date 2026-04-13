Entities
========

The ``datamint.entities`` module provides the core data structures that represent
various objects within the DataMint ecosystem. These entities are built using `Pydantic <https://pydantic-docs.helpmanual.io/>`_ models, ensuring robust data validation,
type safety, and seamless serialization/deserialization when interacting with the DataMint API.

In most user code, prefer working with entity instances instead of passing raw IDs
between endpoint calls. :py:class:`~datamint.entities.Project`,
:py:class:`~datamint.entities.Resource`, and
:py:class:`~datamint.entities.Annotation` can usually be passed directly back into
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
+++++++++++++++++

.. code-block:: python

   annotation = resource.fetch_annotations()[0]

   annotation_data = annotation.fetch_file_data(use_cache=True)
   source_resource = annotation.resource

   print(annotation.name, source_resource.filename)

Reference
---------

.. automodule:: datamint.entities
   :members:
   :undoc-members:
   :show-inheritance: