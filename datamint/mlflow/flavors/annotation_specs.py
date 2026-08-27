"""Derive AnnotationSpec lists from a trained dataset's actual labels. """

from typing import TYPE_CHECKING

from datamint.entities.annotations.annotation_spec import AnnotationSpec, CategoryAnnotationSpec
from datamint.entities.annotations.types import AnnotationType

from .task_type import TaskType

if TYPE_CHECKING:
    from datamint.dataset.base import DatamintBaseDataset


def build_segmentation_annotation_specs(dataset: 'DatamintBaseDataset',
                                        scope: str = 'image') -> list[AnnotationSpec]:
    """Build one AnnotationSpec per segmentation label in the dataset."""
    return [
        AnnotationSpec(type=AnnotationType.SEGMENTATION, scope=scope, identifier=name, required=False)
        for name in dataset.seglabel_list
    ]


def build_detection_annotation_specs(dataset: 'DatamintBaseDataset') -> list[AnnotationSpec]:
    """Build one AnnotationSpec per box-annotation class in the dataset."""
    return [
        AnnotationSpec(type=AnnotationType.SQUARE, scope='image', identifier=name, required=False)
        for name in dataset.box_labels_set
    ]


def build_classification_annotation_specs(dataset: 'DatamintBaseDataset') -> list[CategoryAnnotationSpec]:
    """Build one CategoryAnnotationSpec per classification identifier in the dataset."""
    groups: dict[str, list[str]] = {}
    for identifier, value in dataset.image_categories_set:
        groups.setdefault(identifier, []).append(value)
    return [
        CategoryAnnotationSpec(
            type=AnnotationType.CATEGORY,
            scope='image',
            identifier=ident,
            required=True,
            values=sorted(vals),
        )
        for ident, vals in groups.items()
    ]


def build_annotation_specs_for_task(task_type: 'TaskType | str | None',
                                    dataset: 'DatamintBaseDataset') -> list[AnnotationSpec] | None:
    """Derive annotation specs from a dataset given a (possibly unknown) task type.

    Returns ``None`` -- rather than raising -- when ``task_type`` is missing or
    unrecognized.
    """
    if task_type is None:
        return None
    if isinstance(task_type, str):
        try:
            task_type = TaskType(task_type)
        except ValueError:
            return None

    if task_type in (TaskType.IMAGE_CLASSIFICATION, TaskType.MULTILABEL_IMAGE_CLASSIFICATION):
        return build_classification_annotation_specs(dataset)
    if task_type == TaskType.IMAGE_SEGMENTATION:
        return build_segmentation_annotation_specs(dataset, scope='image')
    if task_type == TaskType.VOLUME_SEGMENTATION:
        return build_segmentation_annotation_specs(dataset, scope='volume')
    if task_type in (TaskType.OBJECT_DETECTION, TaskType.INSTANCE_SEGMENTATION):
        return build_detection_annotation_specs(dataset)

    return None
