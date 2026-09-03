"""API handler for serving-pod log endpoints (MLflow DataMint server)."""
import json
import logging
from collections.abc import Generator
from typing import Any

import httpx

from datamint.entities.pod import ModelPodLogs, PodSummary
from datamint.exceptions import ItemNotFoundError, ValidationError

from ..entity_base_api import ApiConfig, EntityBaseApi

logger = logging.getLogger(__name__)

#: SSE event name signalling that the stream was closed by the server
#: (pod removed or max stream duration reached).
_STREAM_END_EVENT = 'end'


class PodLogsApi(EntityBaseApi[ModelPodLogs]):
    """API handler for serving-pod log endpoints.

    Provides access to recent logs of the Podman pod serving a deployed
    model, listing of the pods serving a model, and live log streaming
    via Server-Sent Events (SSE).

    Note:
        Logs are only available while the pod exists (stopped pods
        included); removed pods lose their logs.
    """

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None) -> None:
        super().__init__(config, ModelPodLogs, 'datamint/api/v1/pod-logs', client)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pod_path(self, model_name: str, add_path: str = '') -> str:
        """Build the request path for a model's pod endpoints.

        Unlike other entity endpoints, pods are keyed by model name
        (not a UUID), so plain ``_make_request`` is used instead of the
        UUID-validating entity request helpers.
        """
        path = f'/{self.endpoint_base}/{model_name}'
        if add_path:
            path += f'/{add_path.strip("/")}'
        return path

    def _raise_if_not_found(self, e: Exception, model_name: str, tag: str) -> None:
        """Re-raise as :class:`ItemNotFoundError` if *e* stems from a 404 response.

        ``BaseApi._make_request`` converts 404s into ``ValidationError`` or a
        bare ``ItemNotFoundError('unknown', {})`` depending on the response
        body, so this helper normalizes both cases. Errors that do not stem
        from a 404 are left untouched (the caller re-raises them as-is).
        """
        if isinstance(e, ItemNotFoundError):
            raise ItemNotFoundError(
                'pod-logs', {'model_name': model_name, 'tag': tag}) from e
        if isinstance(e, ValidationError):
            cause = e.__cause__
            if isinstance(cause, httpx.HTTPStatusError) and cause.response.status_code == 404:
                raise ItemNotFoundError(
                    'pod-logs', {'model_name': model_name, 'tag': tag}) from e

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def get_logs(self,
                 model_name: str,
                 *,
                 tag: str = 'champion',
                 tail: int = 200,
                 since: str | None = None) -> ModelPodLogs:
        """Fetch recent logs from the pod serving *model_name*.

        Args:
            model_name: Name of the registered (deployed) model.
            tag: Image tag of the model pod (e.g. ``'champion'``, ``'challenger'``).
            tail: Maximum number of log lines to return (server-enforced max 5000).
            since: Only return logs after this ISO-8601 timestamp.

        Returns:
            A :class:`ModelPodLogs` with the pod metadata and log lines.

        Raises:
            ItemNotFoundError: If no pod exists for the given model/tag.
        """
        params: dict[str, Any] = {'tag': tag, 'tail': tail}
        if since is not None:
            params['since'] = since
        try:
            response = self._make_request('GET', self._pod_path(model_name), params=params)
        except (ItemNotFoundError, ValidationError) as e:
            self._raise_if_not_found(e, model_name, tag)
            raise
        return self._init_entity_obj(**response.json())

    def list_pods(self,
                  model_name: str,
                  *,
                  tag: str = 'champion') -> list[PodSummary]:
        """List the running and stopped pods serving *model_name*.

        Useful to disambiguate between multiple pods (e.g. champion and
        challenger deployments) before fetching or streaming logs.

        Args:
            model_name: Name of the registered (deployed) model.
            tag: Image tag of the model pods.

        Returns:
            A list of :class:`PodSummary` describing each pod.
        """
        try:
            response = self._make_request(
                'GET', self._pod_path(model_name, 'pods'), params={'tag': tag})
        except (ItemNotFoundError, ValidationError) as e:
            self._raise_if_not_found(e, model_name, tag)
            raise
        return [PodSummary(**pod_data) for pod_data in response.json()]

    def stream_logs(self,
                    model_name: str,
                    *,
                    tag: str = 'champion',
                    interval: float = 2.0) -> Generator[str, None, None]:
        """Stream new log lines from the serving pod via Server-Sent Events.

        The generator ends when the server closes the stream with the
        ``end`` event (pod removed or max stream duration reached).

        Args:
            model_name: Name of the registered (deployed) model.
            tag: Image tag of the model pod.
            interval: Seconds between server-side log polls (0.5 to 10.0).

        Yields:
            Each new log line as a string.
        """
        params: dict[str, Any] = {'tag': tag, 'interval': interval}
        with self._stream_request('GET', self._pod_path(model_name, 'stream'), params=params) as resp:
            for line in resp.iter_lines():
                if not line.startswith('data:'):
                    continue
                payload = line[len('data:'):].strip()
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    # Bare-text payload: treat the whole line as a log line
                    yield payload
                    continue
                event_name = event.get('event')
                if event_name == _STREAM_END_EVENT:
                    return
                log_line = event.get('line', event.get('log'))
                if log_line is not None:
                    yield str(log_line)
