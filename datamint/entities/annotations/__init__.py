from .annotation import Annotation, _normalize_annotation_data
from .box_annotation import BoxAnnotation
from .geometry import BoxGeometry, CoordinateSystem, Geometry, LineGeometry, PointGeometry, RegionGeometry
from .image_classification import ImageClassification
from .image_segmentation import ImageSegmentation
from .line_annotation import LineAnnotation
from .numeric_annotation import NumericAnnotation
from .point_annotation import PointAnnotation
from .region_annotation import RegionAnnotation
from .types import AnnotationType
from .volume_segmentation import VolumeSegmentation


def annotation_from_dict(data: dict) -> Annotation:
    """Factory: map a raw annotation dict to the appropriate Annotation subclass.

    Dispatches on ``annotation_type``:

    * ``'segmentation'`` with a ``class_map`` → :class:`VolumeSegmentation`
    * ``'segmentation'`` without ``class_map`` → :class:`ImageSegmentation`
    * ``'category'`` → :class:`ImageClassification`
    * ``'integer'``/``'float'`` → :class:`NumericAnnotation`
    * ``'line'`` → :class:`LineAnnotation`
    * ``'square'`` → :class:`BoxAnnotation`
    * ``'point'`` → :class:`PointAnnotation`
    * ``'region'`` → :class:`RegionAnnotation`
    * anything else → :class:`Annotation`

    ``segmentation_data`` dicts are automatically deserialised by the
    Pydantic ``BeforeValidator`` defined on
    :class:`~datamint.entities.annotations.base_segmentation.BaseSegmentationAnnotation`.
    ``class_map`` string keys (produced by JSON serialisation) are
    coerced to ``int`` by Pydantic's lax validation.

    Args:
        data: Raw annotation dict as returned by the API.

    Returns:
        A concrete :class:`Annotation` subclass instance.
    """
    normalized_data = _normalize_annotation_data(data)
    annotation_type = normalized_data.get('annotation_type', '')

    if annotation_type in (AnnotationType.SEGMENTATION, AnnotationType.SEGMENTATION.value):
        if normalized_data.get('class_map') is not None:
            return VolumeSegmentation(**normalized_data)
        return ImageSegmentation(**normalized_data)

    if annotation_type in (AnnotationType.CATEGORY, AnnotationType.CATEGORY.value):
        return ImageClassification(**normalized_data)

    if annotation_type in (AnnotationType.INTEGER, AnnotationType.FLOAT):
        return NumericAnnotation(**normalized_data)

    if annotation_type in (AnnotationType.LINE, AnnotationType.LINE.value):
        return LineAnnotation(**normalized_data)

    if annotation_type in (AnnotationType.SQUARE, AnnotationType.SQUARE.value):
        return BoxAnnotation(**normalized_data)

    if annotation_type in (AnnotationType.POINT, AnnotationType.POINT.value):
        return PointAnnotation(**normalized_data)

    if annotation_type in (AnnotationType.REGION, AnnotationType.REGION.value):
        return RegionAnnotation(**normalized_data)

    return Annotation(**normalized_data)


__all__ = [
    "Annotation",
    "AnnotationType",
    "BoxAnnotation",
    "BoxGeometry",
    "CoordinateSystem",
    "Geometry",
    "ImageClassification",
    "ImageSegmentation",
    "LineAnnotation",
    "LineGeometry",
    "NumericAnnotation",
    "PointAnnotation",
    "PointGeometry",
    "RegionAnnotation",
    "RegionGeometry",
    "VolumeSegmentation",
    "annotation_from_dict",
]
