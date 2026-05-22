Annotations Subpackage
======================

The ``datamint.entities.annotations`` subpackage contains all annotation-related
entity classes for the Datamint platform.

.. automodule:: datamint.entities.annotations
   :members:
   :undoc-members:
   :show-inheritance:

Annotation Types
----------------

.. autoclass:: datamint.entities.annotations.AnnotationType
   :members:
   :undoc-members:
   :no-index:

Base Annotation
---------------

.. automodule:: datamint.entities.annotations.annotation
   :members: Annotation
   :undoc-members:
   :show-inheritance:
   :no-index:

Annotation Specification
------------------------

.. automodule:: datamint.entities.annotations.annotation_spec
   :members: AnnotationSpec
   :undoc-members:
   :show-inheritance:
   :no-index:

Segmentation Annotations
------------------------

.. automodule:: datamint.entities.annotations.base_segmentation
   :members: BaseSegmentationAnnotation
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: datamint.entities.annotations.image_segmentation
   :members: ImageSegmentation
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: datamint.entities.annotations.volume_segmentation
   :members: VolumeSegmentation
   :undoc-members:
   :show-inheritance:
   :no-index:

Geometry Annotations
--------------------

.. automodule:: datamint.entities.annotations.base_geometry
   :members: BaseGeometryAnnotation
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: datamint.entities.annotations.box_annotation
   :members: BoxAnnotation, BoxGeometry
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: datamint.entities.annotations.line_annotation
   :members: LineAnnotation, LineGeometry
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: datamint.entities.annotations.geometry
   :members: Geometry, CoordinateSystem
   :undoc-members:
   :show-inheritance:
   :no-index:

Classification Annotations
--------------------------

.. automodule:: datamint.entities.annotations.image_classification
   :members: ImageClassification
   :undoc-members:
   :show-inheritance:
   :no-index:

Annotation Types
----------------

.. automodule:: datamint.entities.annotations.types
   :members:
   :undoc-members:
   :no-index:

Factory Function
----------------

.. automodule:: datamint.entities.annotations
   :members: annotation_from_dict
   :undoc-members:
   :no-index:
