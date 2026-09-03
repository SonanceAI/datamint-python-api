from __future__ import annotations

import logging
from typing import Any

from pydantic import model_validator

from datamint._repr_utils import render_html_card, render_text_block
from datamint.entities.base_entity import BaseEntity

_LOGGER = logging.getLogger(__name__)


class PodSummary(BaseEntity):
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

    @model_validator(mode='before')
    @classmethod
    def _inject_id(cls, data: Any) -> Any:
        """Satisfy the ``id`` requirement of ``BaseEntityModel`` from ``pod_id``."""
        if isinstance(data, dict) and 'id' not in data and 'pod_id' in data:
            data = {**data, 'id': data['pod_id']}
        return data

    def __str__(self) -> str:
        running = 'running' if self.is_running else 'stopped'
        return f'{self.name} ({self.status}, {running})'


class ModelPodLogs(BaseEntity):
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

    @model_validator(mode='before')
    @classmethod
    def _inject_id(cls, data: Any) -> Any:
        """Satisfy the ``id`` requirement of ``BaseEntityModel`` from ``pod_id``."""
        if isinstance(data, dict) and 'id' not in data and 'pod_id' in data:
            data = {**data, 'id': data['pod_id']}
        return data

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
            _render_log_block(shown_logs)

    def __str__(self) -> str:
        header = f'Pod logs — {self.model_name} ({self.pod_name}, {self.status})'
        return render_text_block(header, []) + '\n' + self.text


def _render_log_block(lines: list[str]) -> str:
    """Render a scrollable monospace HTML block with the given log lines."""
    from jinja2 import Environment

    template_source = """\
<div style="max-width: 720px; margin: 10px 0; border-radius: 14px;
           border: 1px solid var(--vscode-panel-border, #d0d7de);
           background: var(--vscode-textCodeBlock-background, #f6f8fa); padding: 14px 16px;">
  <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
             color: var(--vscode-descriptionForeground, #57606a); margin-bottom: 8px;">Log output</div>
  <pre style="margin: 0; max-height: 420px; overflow: auto; font-size: 12px; line-height: 1.55;
             font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
             color: var(--vscode-textPreformat-foreground, var(--vscode-foreground, #1f2328));
             white-space: pre-wrap; word-break: break-word;">{{ log_text }}</pre>
</div>
"""
    env = Environment(autoescape=True)
    return env.from_string(template_source).render(log_text='\n'.join(lines))
