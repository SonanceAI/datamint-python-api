"""
Task type enumeration for Datamint MLflow models.
"""

from enum import Enum


class TaskType(str, Enum):
    """Medical-AI task categories for Datamint models.

    The ``str`` mixin ensures values are JSON-serialisable in MLflow
    metadata without explicit ``.value`` calls.
    """

    # 2D image tasks
    IMAGE_CLASSIFICATION = "image_classification"
    MULTILABEL_IMAGE_CLASSIFICATION = "multilabel_image_classification"
    IMAGE_SEGMENTATION = "image_segmentation"       # semantic, 2D
    INSTANCE_SEGMENTATION = "instance_segmentation"
    OBJECT_DETECTION = "object_detection"

    # 3D/volumetric tasks
    VOLUME_SEGMENTATION = "volume_segmentation"     # semantic, 3D
    VOLUME_CLASSIFICATION = "volume_classification"

    # Video/temporal tasks
    VIDEO_FRAME_CLASSIFICATION = "video_frame_classification"
    VIDEO_SEGMENTATION = "video_segmentation"

    # Medical-specific
    LANDMARK_DETECTION = "landmark_detection"       # anatomical keypoints
    ANOMALY_DETECTION = "anomaly_detection"
    REPORT_GENERATION = "report_generation"         # clinical text output
