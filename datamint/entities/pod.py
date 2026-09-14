from __future__ import annotations

from typing import Any

from pydantic import model_validator

from datamint._repr_utils import render_html_card, render_log_block, render_text_block
from datamint.entities.base_entity import BaseEntity


class _PodIdMixin:
    """Injects ``id`` from ``pod_id`` to satisfy ``BaseEntityModel``."""

    @model_validator(mode='before')
    @classmethod
    def _inject_id(cls, data: Any) -> Any:
        """Satisfy the ``id`` requirement of ``BaseEntityModel`` from ``pod_id``."""
        if isinstance(data, dict) and 'id' not in data and 'pod_id' in data:
            data = {**data, 'id': data['pod_id']}
        return data


class PodSummary(_PodIdMixin, BaseEntity):
    """Summary of a pod serving a model.

    Returned by :meth:`PodLogsApi.list_pods`, which lists both running and
    stopped pods so clients can disambiguate between multiple pods.
    """

    pod_id: str
    name: str
    status: str
    is_running: bool
    host_port: int | None = None
    need_gpu: bool = False

    def __str__(self) -> str:
        running = 'running' if self.is_running else 'stopped'
        return f'{self.name} ({self.status}, {running})'


class ModelPodLogs(_PodIdMixin, BaseEntity):
    """Recent logs from the Podman pod serving a model.

    Returned by :meth:`PodLogsApi.get_logs`. Logs are only available while
    the pod exists (stopped pods included); removed pods lose their logs.
    """

    model_name: str
    image_tag: str
    pod_id: str
    pod_name: str
    status: str
    host_port: int | None = None
    logs: list[str] = []

    @property
    def text(self) -> str:
        """The full log output as a single newline-joined string."""
        return '\n'.join(self.logs)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter Notebooks."""
        max_lines = 200
        shown_logs = self.logs[-max_lines:]
        fields: list[tuple[str, str]] = [
            ('Pod', self.pod_name),
            ('Status', self.status),
            ('Image tag', self.image_tag),
        ]
        if self.host_port is not None:
            fields.append(('Host port', str(self.host_port)))
        fields.append(('Lines', f'{len(self.logs)} (showing last {len(shown_logs)})'))
        return render_html_card(kind='Pod Logs', name=self.model_name, fields=fields) + \
            render_log_block(shown_logs)

    def __str__(self) -> str:
        header = f'Pod logs — {self.model_name} ({self.pod_name}, {self.status})'
        return f'{header}\n{self.text}'
