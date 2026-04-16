from .image_classification import ImageClassification
from .image_segmentation import ImageSegmentation
from .annotation import Annotation, _normalize_annotation_data
from .box_annotation import BoxAnnotation
from .geometry import BoxGeometry, CoordinateSystem, Geometry, LineGeometry
from .line_annotation import LineAnnotation
from .volume_segmentation import VolumeSegmentation
from .types import AnnotationType


def annotation_from_dict(data: dict) -> Annotation:
    """Factory: map a raw annotation dict to the appropriate Annotation subclass.

    Dispatches on ``annotation_type``:

    * ``'segmentation'`` with a ``class_map`` → :class:`VolumeSegmentation`
    * ``'segmentation'`` without ``class_map`` → :class:`ImageSegmentation`
    * anything else → :class:`Annotation`

    ``segmentation_data`` dicts are automatically deserialised by the
    Pydantic ``BeforeValidator`` defined on
    :class:`~.base_segmentation.BaseSegmentationAnnotation`.
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

    if annotation_type in (AnnotationType.LINE, AnnotationType.LINE.value):
        return LineAnnotation(**normalized_data)

    if annotation_type in (AnnotationType.SQUARE, AnnotationType.SQUARE.value):
        return BoxAnnotation(**normalized_data)

    return Annotation(**normalized_data)


__all__ = [
    "BoxAnnotation",
    "BoxGeometry",
    "ImageClassification",
    "ImageSegmentation",
    "Annotation",
    "CoordinateSystem",
    "Geometry",
    "LineAnnotation",
    "LineGeometry",
    "VolumeSegmentation",
    "AnnotationType",
    "annotation_from_dict",
]
