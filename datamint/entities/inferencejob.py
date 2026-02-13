from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections.abc import Callable

from datamint.entities.base_entity import BaseEntity, MISSING_FIELD

if TYPE_CHECKING:
    from datamint.api.endpoints.inference_api import InferenceApi


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
