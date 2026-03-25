from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

from datamint.entities.base_entity import BaseEntity, MISSING_FIELD
from datamint.entities.annotations import annotation_from_dict

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure
    from datamint.api.endpoints.inference_api import InferenceApi
    from datamint.entities.annotations import Annotation

_LOGGER = logging.getLogger(__name__)


class InferenceJob(BaseEntity):
    """Entity representing an inference job."""

    id: str
    status: str
    model_name: str
    resource_id: str | None = None
    frame_idx: int | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    progress_percentage: int = 0
    current_step: str | None = None
    error_message: str | None = None
    save_results: bool = True
    result_data: dict[str, Any] | None = None
    annotation_ids: list | None = None
    recent_logs: list[str] | None = None

    @property
    def is_finished(self) -> bool:
        """Whether the job has reached a terminal state."""
        return self.status.lower() in {'completed', 'failed', 'cancelled', 'error'}
    
    @property
    def predictions(self) -> 'list[list[Annotation]] | None':
        """
        Returns a list of annotations resulting from this inference job, if available.

        Each element of the outer list corresponds to one input resource;
        the inner list contains the annotations produced for that resource.

        Returns:
            ``list[list[Annotation]]`` (one outer list for each input resource) or ``None`` when no predictions are
            stored in :attr:`result_data`.
        """
        if self.result_data and 'predictions' in self.result_data:
            return [
                [annotation_from_dict(ann) for ann in group]
                for group in self.result_data['predictions']
            ]
        return None

    def wait(
        self,
        *,
        on_status: Callable[[InferenceJob], None] | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> InferenceJob:
        """Block until this job reaches a terminal state.

        Uses the SSE stream when available, falling back to polling.

        Args:
            on_status: Optional callback invoked with an updated
                ``InferenceJob`` on every status change.
            poll_interval: Seconds between polls in polling-fallback mode.
            timeout: Maximum seconds to wait.  Raises ``TimeoutError``
                on expiry.

        Returns:
            ``self``, updated in-place with the final status fields.
        """
        api: InferenceApi = self._api  # type: ignore[assignment]

        def _sync_self(updated: InferenceJob) -> None:
            """Copy fields from *updated* into *self*."""
            for field_name, field_value in updated.model_dump().items():
                if field_value != MISSING_FIELD:
                    setattr(self, field_name, field_value)
            if on_status is not None:
                on_status(self)

        api.wait(self.id, on_status=_sync_self, poll_interval=poll_interval, timeout=timeout)
        return self
