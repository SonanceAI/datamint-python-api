from .annotation import Annotation
from datamint.api.dto import AnnotationType


class ImageClassification(Annotation):
    def __init__(self,
                 name: str,
                 value: str,
                 confiability: float = 1.0):
        super().__init__(identifier=name, text_value=value, scope='image',
                         confiability=confiability,
                         annotation_type=AnnotationType.CATEGORY)
