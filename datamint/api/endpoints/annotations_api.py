from typing import Any, Sequence, Literal
import httpx
from datetime import date

from ..base_api import EntityBaseApi, ApiConfig
from datamint.entities.annotation import Annotation
from datamint.entities.resource import Resource
from datamint.apihandler.dto.annotation_dto import AnnotationType


class AnnotationsApi(EntityBaseApi[Annotation]):
    """API handler for annotation-related endpoints."""

    def __init__(self, config: ApiConfig, client: httpx.Client | None = None) -> None:
        """Initialize the annotations API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        super().__init__(config, Annotation, 'annotations', client)

    # def create(self, annotation_data: dict[str, Any]) -> str:
    #     """Create a new annotation.

    #     Args:
    #         annotation_data: Dictionary payload for the annotation.

    #     Returns:
    #         The id of the created annotation.
    #     """
    #     return self._create(annotation_data)

    def get_list(self,
                 resource: str | Resource | None = None,
                 annotation_type: AnnotationType | str | None = None,
                 annotator_email: str | None = None,
                 date_from: date | None = None,
                 date_to: date | None = None,
                 dataset_id: str | None = None,
                 worklist_id: str | None = None,
                 status: Literal['new', 'published'] | None = None,
                 load_ai_segmentations: bool | None = None,
                 ) -> Sequence[Annotation]:
        payload = {
            'resource_id': resource.id if isinstance(resource, Resource) else resource,
            'annotation_type': annotation_type,
            'annotatorEmail': annotator_email,
            'from': date_from.isoformat() if date_from is not None else None,
            'to': date_to.isoformat() if date_to is not None else None,
            'dataset_id': dataset_id,
            'annotation_worklist_id': worklist_id,
            'status': status,
            'load_ai_segmentations': load_ai_segmentations
        }

        # remove nones
        payload = {k: v for k, v in payload.items() if v is not None}
        return super().get_list(**payload)
