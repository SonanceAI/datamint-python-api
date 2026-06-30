"""API handler for model deployment endpoints."""
from typing import Any, Literal
from collections.abc import Callable, Generator
import json
import logging
import time

import httpx

from datamint.exceptions import ResourceNotFoundError, JobTimeoutError
from ..entity_base_api import EntityBaseApi, ApiConfig
from datamint.entities.deployjob import DeployJob

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({'completed', 'failed', 'cancelled', 'error'})

class DeployModelApi(EntityBaseApi[DeployJob]):
    """API handler for model deployment endpoints."""

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None) -> None:
        super().__init__(config, DeployJob, 'datamint/api/v1/deploy-model', client)

    def get_by_id(self, entity_id: str) -> DeployJob:
        """Get deployment job status by ID."""
        response = self._make_request('GET', f'/{self.endpoint_base}/status/{entity_id}')
        data = response.json()
        if 'job_id' in data:
            data['id'] = data.pop('job_id')
        self._validate_uuid(data['id'])
        try:
            return self._init_entity_obj(**data)
        except ResourceNotFoundError as e:
            e.resource_type = 'DeployJob'
            e.params = {'id': entity_id}
            raise

    def stream_status(self, job_id: str) -> Generator[dict[str, Any], None, None]:
        """Stream status updates for a deployment job via Server-Sent Events.

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
        job: str | DeployJob,
        *,
        on_status: Callable[[DeployJob], None] | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = 1800,
    ) -> None:
        """Block until a deployment job reaches a terminal state.

        First attempts to follow the SSE stream. If the stream is
        unavailable or drops early the method falls back to polling
        ``get_by_id`` at *poll_interval* seconds.

        Args:
            job: Job ID string or ``DeployJob`` entity. In-place updates to the provided ``DeployJob`` are made on every status change.
            on_status: Optional callback invoked with an updated
                ``DeployJob`` each time a status update is received.
            poll_interval: Seconds between polls when falling back to
                polling mode. Default ``2.0``.
            timeout: Maximum seconds to wait. ``None`` means wait
                indefinitely. Raises ``TimeoutError`` on expiry.

        Raises:
            TimeoutError: If *timeout* is set and the job has not
                finished within that duration.
        """
        job_id = self._entid(job)
        deadline = (time.monotonic() + timeout) if timeout is not None else None

        def _check_timeout() -> None:
            if deadline is not None and time.monotonic() >= deadline:
                raise JobTimeoutError(f"Deployment job {job_id} did not finish within {timeout}s")

        def _notify(event: dict) -> None:
            if on_status is None:
                return
            if isinstance(job, DeployJob):
                # SSE events are partial updates — apply known fields in-place
                for key, value in event.items():
                    try:
                        setattr(job, key, value)
                    except Exception:
                        pass
                on_status(job)
            else:
                on_status(self.get_by_id(job_id))

        # --- Try SSE stream first ---
        try:
            for event in self.stream_status(job_id):
                _check_timeout()
                _notify(event)
                if event.get('status', '').lower() in _TERMINAL_STATUSES:
                    return
        except Exception as e:
            logger.warning(f"SSE stream ended or failed ({e}); falling back to polling")

        # --- Polling fallback ---
        while True:
            _check_timeout()
            current_job = self.get_by_id(job_id)
            if on_status is not None:
                on_status(current_job)
            if current_job.status.lower() in _TERMINAL_STATUSES:
                return
            time.sleep(poll_interval)

    def start(self,
              model_name: str,
              model_version: int | None = None,
              model_alias: str | None = None,
              image_name: str | None = None,
              with_gpu: bool = False,
              convert_to_onnx: bool = False,
              input_shape: list[int] | None = None) -> DeployJob:
        """Start a new deployment job."""
        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "model_alias": model_alias,
            "image_name": image_name,
            "with_gpu": with_gpu,
            "convert_to_onnx": convert_to_onnx,
            "input_shape": input_shape
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        response = self._make_request('POST', f'/{self.endpoint_base}/start', json=payload)
        data = response.json()
        return self.get_by_id(data['job_id'])

    def cancel(self, job: str | DeployJob) -> bool:
        """Cancel a deployment job."""
        job_id = self._entid(job)
        response = self._make_request('POST', f'/{self.endpoint_base}/cancel/{job_id}')
        return response.json().get('success', False)

    def list_active_jobs(self) -> dict:
        """List active deployment jobs count."""
        response = self._make_request('GET', f'/{self.endpoint_base}/jobs')
        return response.json()

    def list_images(self, model_name: str | None = None) -> list[dict]:
        """List deployed model images."""
        params = {}
        if model_name:
            params['model_name'] = model_name
        response = self._make_request('GET', f'/{self.endpoint_base}/images', params=params)
        return response.json()

    def remove_image(self, model_name: str, tag: str | None = None) -> dict:
        """Remove a deployed model image."""
        params = {}
        if tag:
            params['tag'] = tag
        response = self._make_request('DELETE', f'/{self.endpoint_base}/image/{model_name}', params=params)
        return response.json()

    def image_exists(self, model_name: str, tag: str = "champion") -> bool:
        """Check if a model image exists."""
        params = {'tag': tag}
        response = self._make_request('GET', f'/{self.endpoint_base}/image/{model_name}/exists', params=params)
        return response.json().get('exists', False)

