"""API handler for model inference endpoints (MLflow DataMint server)."""
from typing import Any, Literal
from collections.abc import Callable, Generator
import json
import logging
import time

import httpx

from ..entity_base_api import EntityBaseApi, ApiConfig
from datamint.entities.inferencejob import InferenceJob

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({'completed', 'failed', 'cancelled', 'error'})


class InferenceApi(EntityBaseApi[InferenceJob]):
    """API handler for model inference endpoints.

    Provides methods to submit inference jobs, poll their status,
    cancel running jobs, and use specialised prediction endpoints
    (image, frame, slice, volume).
    """

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None) -> None:
        super().__init__(config, InferenceJob, 'datamint/api/v1/model-inference', client)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_job_response(self, data: dict) -> InferenceJob:
        """Normalise a job-status response into an ``InferenceJob`` entity."""
        if 'job_id' in data:
            data['id'] = data.pop('job_id')
        return self._init_entity_obj(**data)

    @staticmethod
    def _build_common_payload(
        model_name: str,
        model_version: int | None = None,
        model_alias: str | None = None,
        resource_id: str | None = None,
        file_path: str | None = None,
        save_results: bool = False,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the payload keys shared by every inference request."""
        payload: dict[str, Any] = {"model_name": model_name}
        if model_version is not None:
            payload["model_version"] = model_version
        if model_alias is not None:
            payload["model_alias"] = model_alias
        if resource_id is not None:
            payload["resource_id"] = resource_id
        if file_path is not None:
            payload["file_path"] = file_path
        if save_results:
            payload["save_results"] = save_results
        if params:
            payload["params"] = params
        return payload

    # ------------------------------------------------------------------
    # Generic inference
    # ------------------------------------------------------------------

    def submit(
        self,
        model_name: str,
        *,
        model_version: int | None = None,
        model_alias: str | None = None,
        resource_id: str | None = None,
        resource_ids: list[str] | None = None,
        file_path: str | None = None,
        file_paths: list[str] | None = None,
        save_results: bool = False,
        params: dict[str, Any] | None = None,
    ) -> InferenceJob:
        """Submit an inference job for background processing.

        Args:
            model_name: Name of the registered model.
            model_version: Specific model version number.
            model_alias: Model alias (e.g. ``'champion'``).
            resource_id: Single resource ID from DataMint API.
            resource_ids: List of resource IDs.
            file_path: Local file path.
            file_paths: List of local file paths.
            save_results: Whether to save results to the API.
            params: Additional parameters forwarded to the model.

        Returns:
            The created ``InferenceJob`` (with initial status).
        """
        payload = self._build_common_payload(
            model_name,
            model_version=model_version,
            model_alias=model_alias,
            resource_id=resource_id,
            file_path=file_path,
            save_results=save_results,
            params=params,
        )
        if resource_ids is not None:
            payload["resource_ids"] = resource_ids
        if file_paths is not None:
            payload["file_paths"] = file_paths

        response = self._make_request('POST', f'/{self.endpoint_base}', json=payload)
        data = response.json()
        return self.get_status(data['job_id'])

    # ------------------------------------------------------------------
    # Status / cancel
    # ------------------------------------------------------------------

    def get_status(self, job_id: str) -> InferenceJob:
        """Get the current status of an inference job.

        Args:
            job_id: The job identifier.

        Returns:
            An ``InferenceJob`` populated with the latest status.
        """
        response = self._make_request('GET', f'/{self.endpoint_base}/status/{job_id}')
        return self._parse_job_response(response.json())

    def get_by_id(self, entity_id: str) -> InferenceJob:
        """Alias for ``get_status`` to satisfy ``EntityBaseApi`` interface."""
        return self.get_status(entity_id)

    def stream_status(self, job_id: str) -> Generator[dict[str, Any], None, None]:
        """Stream status updates for an inference job via Server-Sent Events.

        Yields dictionaries parsed from SSE ``data:`` lines until the
        stream is closed by the server.

        Args:
            job_id: The job identifier.

        Yields:
            Parsed JSON dictionaries for each SSE event.
        """
        with self._stream_request('GET', f'/{self.endpoint_base}/status/{job_id}/stream') as resp:
            for line in resp.iter_lines():
                if line.startswith('data:'):
                    payload = line[len('data:'):].strip()
                    if payload:
                        yield json.loads(payload)

    def wait(
        self,
        job: str | InferenceJob,
        *,
        on_status: Callable[[InferenceJob], None] | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> InferenceJob:
        """Block until an inference job reaches a terminal state.

        First attempts to follow the SSE stream. If the stream is
        unavailable or drops early the method falls back to polling
        ``get_status`` at *poll_interval* seconds.

        Args:
            job: Job ID string or ``InferenceJob`` entity.
            on_status: Optional callback invoked with an updated
                ``InferenceJob`` each time a status update is received.
            poll_interval: Seconds between polls when falling back to
                polling mode. Default ``2.0``.
            timeout: Maximum seconds to wait. ``None`` means wait
                indefinitely. Raises ``TimeoutError`` on expiry.

        Returns:
            The ``InferenceJob`` in its terminal state.

        Raises:
            TimeoutError: If *timeout* is set and the job has not
                finished within that duration.
        """
        job_id = self._entid(job) if not isinstance(job, str) else job
        deadline = (time.monotonic() + timeout) if timeout is not None else None

        def _check_timeout() -> None:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Inference job {job_id} did not finish within {timeout}s"
                )

        # --- Try SSE stream first ---
        try:
            for event in self.stream_status(job_id):
                _check_timeout()
                status_str = event.get('status', '')
                current_job = self._parse_job_response(event)
                if on_status is not None:
                    on_status(current_job)
                if status_str.lower() in _TERMINAL_STATUSES:
                    return current_job
        except Exception as e:
            logger.warning(f"SSE stream ended or failed ({e}); falling back to polling")

        # --- Polling fallback ---
        while True:
            _check_timeout()
            current_job = self.get_status(job_id)
            if on_status is not None:
                on_status(current_job)
            if current_job.status.lower() in _TERMINAL_STATUSES:
                return current_job
            time.sleep(poll_interval)

    def cancel(self, job: str | InferenceJob) -> bool:
        """Cancel a running inference job.

        Args:
            job: Job ID string or ``InferenceJob`` entity.

        Returns:
            ``True`` if the cancellation was acknowledged.
        """
        job_id = self._entid(job)
        response = self._make_request('POST', f'/{self.endpoint_base}/cancel/{job_id}')
        return response.json().get('success', False)

    # ------------------------------------------------------------------
    # Specialised prediction endpoints
    # ------------------------------------------------------------------

    def predict_image(
        self,
        model_name: str,
        *,
        model_version: int | None = None,
        model_alias: str | None = None,
        resource_id: str | None = None,
        file_path: str | None = None,
        save_results: bool = False,
        params: dict[str, Any] | None = None,
    ) -> InferenceJob:
        """Submit an image prediction job.

        Args:
            model_name: Name of the registered model.
            model_version: Specific model version number.
            model_alias: Model alias (e.g. ``'champion'``).
            resource_id: Resource ID from DataMint API.
            file_path: Local file path.
            save_results: Whether to save results.
            params: Additional parameters.

        Returns:
            The created ``InferenceJob``.
        """
        payload = self._build_common_payload(
            model_name,
            model_version=model_version,
            model_alias=model_alias,
            resource_id=resource_id,
            file_path=file_path,
            save_results=save_results,
            params=params,
        )
        response = self._make_request('POST', f'/{self.endpoint_base}/predict-image', json=payload)
        data = response.json()
        return self.get_status(data['job_id'])

    def predict_frame(
        self,
        model_name: str,
        frame_index: int,
        *,
        model_version: int | None = None,
        model_alias: str | None = None,
        resource_id: str | None = None,
        file_path: str | None = None,
        save_results: bool = False,
        params: dict[str, Any] | None = None,
    ) -> InferenceJob:
        """Submit a frame-specific prediction job (for video resources).

        Args:
            model_name: Name of the registered model.
            frame_index: Frame index to process.
            model_version: Specific model version number.
            model_alias: Model alias.
            resource_id: Resource ID from DataMint API.
            file_path: Local file path.
            save_results: Whether to save results.
            params: Additional parameters.

        Returns:
            The created ``InferenceJob``.
        """
        payload = self._build_common_payload(
            model_name,
            model_version=model_version,
            model_alias=model_alias,
            resource_id=resource_id,
            file_path=file_path,
            save_results=save_results,
            params=params,
        )
        payload["frame_index"] = frame_index
        response = self._make_request('POST', f'/{self.endpoint_base}/predict-frame', json=payload)
        data = response.json()
        return self.get_status(data['job_id'])

    def predict_slice(
        self,
        model_name: str,
        slice_index: int,
        axis: Literal['axial', 'sagittal', 'coronal'],
        *,
        model_version: int | None = None,
        model_alias: str | None = None,
        resource_id: str | None = None,
        file_path: str | None = None,
        save_results: bool = False,
        params: dict[str, Any] | None = None,
    ) -> InferenceJob:
        """Submit a slice-specific prediction job for 3D volumes.

        Args:
            model_name: Name of the registered model.
            slice_index: Slice index to process.
            axis: Anatomical axis (``'axial'``, ``'sagittal'``, or ``'coronal'``).
            model_version: Specific model version number.
            model_alias: Model alias.
            resource_id: Resource ID from DataMint API.
            file_path: Local file path.
            save_results: Whether to save results.
            params: Additional parameters.

        Returns:
            The created ``InferenceJob``.
        """
        payload = self._build_common_payload(
            model_name,
            model_version=model_version,
            model_alias=model_alias,
            resource_id=resource_id,
            file_path=file_path,
            save_results=save_results,
            params=params,
        )
        payload["slice_index"] = slice_index
        payload["axis"] = axis
        response = self._make_request('POST', f'/{self.endpoint_base}/predict-slice', json=payload)
        data = response.json()
        return self.get_status(data['job_id'])

    def predict_volume(
        self,
        model_name: str,
        *,
        model_version: int | None = None,
        model_alias: str | None = None,
        resource_id: str | None = None,
        file_path: str | None = None,
        save_results: bool = False,
        params: dict[str, Any] | None = None,
    ) -> InferenceJob:
        """Submit a volume prediction job.

        Args:
            model_name: Name of the registered model.
            model_version: Specific model version number.
            model_alias: Model alias.
            resource_id: Resource ID from DataMint API.
            file_path: Local file path.
            save_results: Whether to save results.
            params: Additional parameters.

        Returns:
            The created ``InferenceJob``.
        """
        payload = self._build_common_payload(
            model_name,
            model_version=model_version,
            model_alias=model_alias,
            resource_id=resource_id,
            file_path=file_path,
            save_results=save_results,
            params=params,
        )
        response = self._make_request('POST', f'/{self.endpoint_base}/predict-volume', json=payload)
        data = response.json()
        return self.get_status(data['job_id'])