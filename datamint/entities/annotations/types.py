import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


class AnnotationType(StrEnum):
    SEGMENTATION = 'segmentation'
    AREA = 'area'
    DISTANCE = 'distance'
    ANGLE = 'angle'
    POINT = 'point'
    LINE = 'line'
    REGION = 'region'
    SQUARE = 'square'
    CIRCLE = 'circle'
    CATEGORY = 'category'
    LABEL = 'label'