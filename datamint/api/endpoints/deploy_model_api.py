"""API handler for model deployment endpoints."""
import json
import logging
import threading
import time
from collections.abc import Callable, Generator
from typing import Any

import httpx

from datamint.configs import DEFAULT_DEPLOY_MODEL_ALIAS
from datamint.entities.buildlog import BuildLogs
from datamint.entities.deployjob import DeployJob
from datamint.exceptions import ItemNotFoundError, JobTimeoutError, ResourceNotFoundError, ValidationError

from ..entity_base_api import ApiConfig, EntityBaseApi

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({'completed', 'failed', 'cancelled', 'error'})

#: SSE event name carrying a single new build-log line
_LOG_EVENT = 'log'
#: SSE event name signalling that the log stream was closed by the server
#: (job completed/failed/cancelled, job not found or timeout).
_STREAM_END_EVENT = 'end'

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

    def stream_status(self,
                      job: str | DeployJob | None = None) -> Generator[dict[str, Any], None, None]:
        """Stream status updates for a deployment job via Server-Sent Events.

        Yields dictionaries parsed from SSE ``data:`` lines until the
        stream is closed by the server.

        Args:
            job: The job ID string or ``DeployJob`` instance.

        Yields:
            Parsed JSON dictionaries for each SSE event.
        """
        if job is None:
            raise TypeError("stream_status() missing required argument: 'job'")
        job_id_str = self._entid(job)

        with self._stream_request('GET', f'/{self.endpoint_base}/status/{job_id_str}/stream') as resp:
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
        on_log: Callable[[str], None] | None = None,
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
            on_log: Optional callback invoked with each new build-log
                line as it streams in. The callback is invoked from a
                background thread, so it must be thread-safe.
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

        # --- Background build-log streaming (best effort) ---
        stop_logs = threading.Event()
        log_thread: threading.Thread | None = None
        if on_log is not None:
            def _consume_logs() -> None:
                try:
                    for line in self.stream_logs(job_id):
                        if stop_logs.is_set():
                            break
                        on_log(line)
                except Exception as e:
                    logger.debug(f"Build-log stream ended or failed: {e}")

            log_thread = threading.Thread(target=_consume_logs,
                                          daemon=True,
                                          name=f'deploy-build-logs-{job_id}')
            log_thread.start()

        try:
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
        finally:
            stop_logs.set()
            if log_thread is not None:
                # The server closes the log stream when the job reaches a
                # terminal state, so the thread normally ends on its own.
                log_thread.join(timeout=5.0)

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

    def image_exists(self, model_name: str, tag: str = DEFAULT_DEPLOY_MODEL_ALIAS) -> bool:
        """Check if a model image exists."""
        params = {'tag': tag}
        response = self._make_request('GET', f'/{self.endpoint_base}/image/{model_name}/exists', params=params)
        return response.json().get('exists', False)

    # ------------------------------------------------------------------
    # Build logs
    # ------------------------------------------------------------------

    def _logs_path(self, job_id: str, add_path: str = '') -> str:
        """Build the request path for a build job's log endpoints."""
        path = f'/{self.endpoint_base}/logs/{job_id}'
        if add_path:
            path += f'/{add_path.strip("/")}'
        return path

    def _raise_if_not_found(self, e: Exception, job_id: str) -> None:
        """Re-raise as :class:`ItemNotFoundError` if *e* stems from a 404 response.

        ``BaseApi._make_request`` converts 404s into ``ValidationError`` or a
        bare ``ItemNotFoundError('unknown', {})`` depending on the response
        body, so this helper normalizes both cases. Errors that do not stem
        from a 404 are left untouched (the caller re-raises them as-is).
        """
        if isinstance(e, ItemNotFoundError):
            raise ItemNotFoundError('deploy-model', {'job_id': job_id}) from e
        if isinstance(e, ValidationError):
            cause = e.__cause__
            if isinstance(cause, httpx.HTTPStatusError) and cause.response.status_code == 404:
                raise ItemNotFoundError('deploy-model', {'job_id': job_id}) from e

    def get_logs(self,
                 job: str | DeployJob,
                 *,
                 tail: int = 2000,
                 since: str | None = None) -> BuildLogs:
        """Fetch the build logs of a Docker build (deployment) job.

        Logs are read from the server's real-time in-memory buffer while the
        job is running, otherwise from the persisted DB snapshot (see
        :attr:`BuildLogs.log_source`). Only the job owner (per customer) can
        read them.

        Args:
            job: Job ID string or ``DeployJob`` entity.
            tail: Maximum number of log lines to return (server-enforced max 5000).
            since: Only return lines logged at/after this ISO-8601 timestamp.

        Returns:
            A :class:`BuildLogs` entity with the job metadata and log lines.

        Raises:
            ValueError: If the job ID is not a valid UUID.
            ItemNotFoundError: If the job does not exist.
            httpx.HTTPStatusError: If the server returns another error status.
        """
        job_id = self._entid(job)
        self._validate_uuid(job_id)
        params: dict[str, Any] = {'tail': tail}
        if since is not None:
            params['since'] = since
        try:
            response = self._make_request('GET', self._logs_path(job_id), params=params)
        except (ItemNotFoundError, ValidationError) as e:
            self._raise_if_not_found(e, job_id)
            raise
        build_logs = BuildLogs(**response.json())
        build_logs._api = self
        return build_logs

    @staticmethod
    def _parse_log_event(payload: str, sse_event_name: str | None) -> dict[str, Any]:
        """Parse an SSE ``data:`` payload into a normalized event dict.

        Standard SSE framing carries the event name on ``event:`` lines; the
        DataMint server also embeds it inside the JSON payload, so both are
        honored (the payload value takes precedence).
        """
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            # Bare-text payload: treat the whole line as a log line
            event = {'line': payload}
        if not isinstance(event, dict):
            # Non-object JSON payload: treat it as a log line
            event = {'line': str(event)}
        if 'event' not in event:
            event['event'] = sse_event_name or _LOG_EVENT
        return event

    def stream_log_events(self,
                          job: str | DeployJob,
                          *,
                          interval: float = 2.0) -> Generator[dict[str, Any], None, None]:
        """Stream build-log SSE events for a Docker build job.

        This is the low-level variant of :meth:`stream_logs`: it yields the
        raw parsed event dictionaries so consumers can react to the
        ``start`` metadata event (job metadata) as well.

        The generator ends when the server closes the stream with the
        ``end`` event (job completed/failed/cancelled, job not found or
        max stream duration reached).

        Args:
            job: Job ID string or ``DeployJob`` entity.
            interval: Seconds between server-side log polls for archived
                (DB-only) jobs (0.5 to 10.0).

        Yields:
            Event dictionaries with at least an ``'event'`` key
            (``'start'``, ``'log'`` or ``'end'``). For ``'log'`` events the
            line is available under ``'line'`` (or ``'log'``).

        Raises:
            ValueError: If the job ID is not a valid UUID.
            ItemNotFoundError: If the job does not exist.
            httpx.HTTPStatusError: If the server returns another error status.
        """
        job_id = self._entid(job)
        self._validate_uuid(job_id)
        params: dict[str, Any] = {'interval': interval}
        with self._stream_request('GET', self._logs_path(job_id, 'stream'), params=params) as resp:
            if resp.status_code == 404:
                raise ItemNotFoundError('deploy-model', {'job_id': job_id})
            resp.raise_for_status()
            sse_event_name: str | None = None
            for line in resp.iter_lines():
                if line.startswith('event:'):
                    sse_event_name = line[len('event:'):].strip()
                    continue
                if not line.startswith('data:'):
                    continue
                payload = line[len('data:'):].strip()
                if not payload:
                    continue
                event = self._parse_log_event(payload, sse_event_name)
                if event.get('event') == _STREAM_END_EVENT:
                    return
                yield event

    def stream_logs(self,
                    job: str | DeployJob,
                    *,
                    interval: float = 2.0) -> Generator[str, None, None]:
        """Stream new build-log lines of a Docker build job via SSE.

        Convenience wrapper over :meth:`stream_log_events` that yields only
        the ``log`` events as plain strings. The generator ends when the
        server closes the stream with the ``end`` event (job completed/
        failed/cancelled, job not found or max stream duration reached).

        Args:
            job: Job ID string or ``DeployJob`` entity.
            interval: Seconds between server-side log polls for archived
                (DB-only) jobs (0.5 to 10.0).

        Yields:
            Each new build-log line as a string.

        Raises:
            ValueError: If the job ID is not a valid UUID.
            ItemNotFoundError: If the job does not exist.
            httpx.HTTPStatusError: If the server returns another error status.
        """
        for event in self.stream_log_events(job, interval=interval):
            if event.get('event') != _LOG_EVENT:
                continue
            line = event.get('line', event.get('log'))
            if line is not None:
                yield str(line)

