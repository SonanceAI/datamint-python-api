from __future__ import annotations

from typing import Any

from pydantic import model_validator

from datamint._repr_utils import render_html_card, render_log_block
from datamint.entities.base_entity import BaseEntity


class _JobIdMixin:
    """Injects ``id`` from ``job_id`` to satisfy ``BaseEntity``."""

    @model_validator(mode='before')
    @classmethod
    def _inject_id(cls, data: Any) -> Any:
        """Satisfy the ``id`` requirement of ``BaseEntity`` from ``job_id``."""
        if isinstance(data, dict) and 'id' not in data and 'job_id' in data:
            data = {**data, 'id': data['job_id']}
        return data


class BuildLogs(_JobIdMixin, BaseEntity):
    """Build logs for a Docker image build (deployment) job.

    Returned by :meth:`DeployModelApi.get_logs`. Logs are read from the
    server's real-time in-memory buffer while the job runs
    (``log_source='memory'``), otherwise from the persisted DB snapshot
    (``log_source='db'``). Only the job owner (per customer) can read them.
    """

    job_id: str
    model_name: str
    job_status: str
    image_name: str | None = None
    image_tag: str | None = None
    total_lines: int = 0
    returned_lines: int = 0
    log_source: str = 'db'
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
            ('Job status', self.job_status),
            ('Log source', self.log_source),
        ]
        if self.image_name is not None:
            fields.append(('Image', self.image_name))
        if self.image_tag is not None:
            fields.append(('Tag', self.image_tag))
        fields.append(('Lines',
                       f'{self.returned_lines} of {self.total_lines}'
                       f' (showing last {len(shown_logs)})'))
        return render_html_card(kind='Build Logs', name=self.model_name, fields=fields) + \
            render_log_block(shown_logs)

    def __str__(self) -> str:
        header = (f'Build logs — {self.model_name}'
                  f' ({self.job_id}, {self.job_status}, source: {self.log_source})')
        return f'{header}\n{self.text}'
