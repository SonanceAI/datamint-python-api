from .image_classification import ImageClassification
from .image_segmentation import ImageSegmentation
from .annotation import Annotation
from .volume_segmentation import VolumeSegmentation
from datamint.api.dto import AnnotationType # FIXME: move this to this module


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
    annotation_type = data.get('annotation_type', '')

    if annotation_type in (AnnotationType.SEGMENTATION, AnnotationType.SEGMENTATION.value):
        if data.get('class_map') is not None:
            return VolumeSegmentation(**data)
        return ImageSegmentation(**data)

    return Annotation(**data)


__all__ = [
    "ImageClassification",
    "ImageSegmentation",
    "Annotation",
    "VolumeSegmentation",
    "AnnotationType",
    "annotation_from_dict",
]
