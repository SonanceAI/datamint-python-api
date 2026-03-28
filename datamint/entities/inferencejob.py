from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any, TYPE_CHECKING

from datamint.entities.base_entity import BaseEntity, MISSING_FIELD
from datamint.entities.annotations import annotation_from_dict

if TYPE_CHECKING:
    from datamint.api.endpoints.inference_api import InferenceApi
    from datamint.entities.annotations import Annotation

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jinja2 HTML template for InferenceJob Jupyter repr
# ---------------------------------------------------------------------------
_INFERENCE_JOB_HTML_TEMPLATE = """\
{%- set status_styles = {
    'completed': ('#166534', 'rgba(34, 197, 94, 0.16)', 'rgba(34, 197, 94, 0.32)'),
    'failed':    ('#b91c1c', 'rgba(239, 68, 68, 0.16)',  'rgba(239, 68, 68, 0.28)'),
    'error':     ('#b91c1c', 'rgba(239, 68, 68, 0.16)',  'rgba(239, 68, 68, 0.28)'),
    'cancelled': ('#475569', 'rgba(148, 163, 184, 0.18)', 'rgba(148, 163, 184, 0.28)'),
} %}
{%- set accent_color, badge_bg, badge_border = status_styles.get(
    status_lower, ('#1d4ed8', 'rgba(59, 130, 246, 0.14)', 'rgba(59, 130, 246, 0.26)')
) %}
<div style="max-width: 760px; margin: 10px 0; overflow: hidden; border-radius: 20px;
           border: 1px solid var(--vscode-panel-border, #d0d7de);
           background: var(--vscode-editor-background, #ffffff);
           color: var(--vscode-foreground, #1f2328);
           box-shadow: 0 14px 38px rgba(15, 23, 42, 0.12);">

  {# ---- Header ---- #}
  <div style="padding: 20px 22px; border-bottom: 1px solid var(--vscode-panel-border, #d0d7de);
             background: linear-gradient(135deg, {{ badge_bg }}, rgba(255,255,255,0));">
    <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap;">
      <div>
        <div style="font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
                   color: var(--vscode-descriptionForeground, #57606a);">Inference Job</div>
        <div style="margin-top: 8px; font-size: 24px; font-weight: 800; line-height: 1.2; color: inherit;">{{ model_name }}</div>
      </div>
      <div style="display: inline-flex; align-items: center; padding: 8px 12px; border-radius: 999px;
                 border: 1px solid {{ badge_border }}; background: {{ badge_bg }}; color: {{ accent_color }};
                 font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;">{{ status }}</div>
    </div>
    {%- if frame_idx is not none %}
    <div style="margin-top: 14px;">
      <span style="display: inline-block; padding: 3px 8px; border-radius: 999px;
                  background: var(--vscode-textCodeBlock-background, #f6f8fa);
                  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
                  font-size: 13px;">frame {{ frame_idx }}</span>
    </div>
    {%- endif %}
  </div>

  <div style="padding: 18px 22px 22px;">

    {# ---- Progress bar ---- #}
    {%- if show_progress %}
    <div style="padding: 16px; margin-bottom: 16px; border: 1px solid var(--vscode-panel-border, #d0d7de);
               border-radius: 14px; background: var(--vscode-textCodeBlock-background, rgba(127,127,127,0.08));">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
        <div style="font-size: 13px; font-weight: 700; color: var(--vscode-descriptionForeground, #57606a);">Progress</div>
        <div style="font-size: 15px; font-weight: 800; color: {{ accent_color }};">{{ progress_value }}%</div>
      </div>
      <div style="height: 10px; overflow: hidden; border-radius: 999px;
                 background: var(--vscode-editor-inactiveSelectionBackground, rgba(148,163,184,0.24));">
        <div style="width: {{ progress_value }}%; {% if progress_value > 0 %}min-width: 10px; {% endif %}height: 100%;
                   border-radius: 999px; background: linear-gradient(90deg, {{ accent_color }}, #38bdf8);"></div>
      </div>
      {%- if current_step %}
      <div style="margin-top: 10px; font-size: 13px; color: var(--vscode-descriptionForeground, #57606a);">{{ current_step }}</div>
      {%- endif %}
    </div>
    {%- endif %}

    {# ---- Metric cards ---- #}
    {%- set ns = namespace(has_metrics=false) %}
    {%- for label, value in metrics %}{% if value %}{% set ns.has_metrics = true %}{% endif %}{% endfor %}
    {%- if ns.has_metrics %}
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;">
      {%- for label, value in metrics %}
      {%- if value %}
      <div style="padding: 14px 16px; border: 1px solid var(--vscode-panel-border, #d0d7de); border-radius: 14px;
                 background: var(--vscode-textCodeBlock-background, rgba(127,127,127,0.08));">
        <div style="font-size: 12px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
                   color: var(--vscode-descriptionForeground, #57606a);">{{ label }}</div>
        <div style="margin-top: 8px; font-size: 14px; line-height: 1.5;">
          <span style="display: inline-block; padding: 3px 8px; border-radius: 999px;
                      background: var(--vscode-textCodeBlock-background, #f6f8fa);
                      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
                      font-size: 13px;">{{ value[:96] }}{% if value | length > 96 %}&hellip;{% endif %}</span>
        </div>
      </div>
      {%- endif %}
      {%- endfor %}
    </div>
    {%- endif %}

    {# ---- Error ---- #}
    {%- if error_message %}
    <div style="margin-top: 16px; padding: 14px 16px; border-radius: 14px;
               border: 1px solid {{ badge_border }}; background: {{ badge_bg }};">
      <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
                 color: {{ accent_color }};">Error</div>
      <div style="margin-top: 8px; font-size: 14px; line-height: 1.5;">{{ error_message }}</div>
    </div>
    {%- endif %}

    {# ---- Recent logs ---- #}
    {%- if recent_logs %}
    <div style="margin-top: 16px; padding: 14px 16px; border: 1px solid var(--vscode-panel-border, #d0d7de);
               border-radius: 14px; background: var(--vscode-textCodeBlock-background, rgba(127,127,127,0.08));">
      <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
                 color: var(--vscode-descriptionForeground, #57606a);">Recent logs</div>
      <div style="margin-top: 8px; font-size: 13px; line-height: 1.6;
                 font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;">
        {%- for line in recent_logs %}
        {{ line }}{% if not loop.last %}<br/>{% endif %}
        {%- endfor %}
      </div>
    </div>
    {%- endif %}

  </div>
</div>
"""

_inference_job_template = None


def _get_inference_job_template():
    """Lazily compile and cache the Jinja2 InferenceJob template."""
    global _inference_job_template
    if _inference_job_template is None:
        from jinja2 import Environment
        _inference_job_template = Environment(autoescape=True).from_string(_INFERENCE_JOB_HTML_TEMPLATE)
    return _inference_job_template


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

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter Notebooks."""
        progress_value = max(0, min(self.progress_percentage, 100))
        return _get_inference_job_template().render(
            status_lower=self.status.lower(),
            status=self.status,
            model_name=self.model_name,
            frame_idx=self.frame_idx,
            progress_value=progress_value,
            show_progress=not self.is_finished or progress_value > 0 or self.current_step,
            current_step=self.current_step,
            metrics=[
                ('Model', self.model_name),
                ('Current step', self.current_step),
                ('Created', self.created_at),
                ('Completed', self.completed_at),
            ],
            error_message=self.error_message,
            recent_logs=self.recent_logs[-3:] if self.recent_logs else None,
        )

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
        In-place updates to this object are made on every status change.

        Args:
            on_status: Optional callback invoked with an updated
                ``InferenceJob`` on every status change.
            poll_interval: Seconds between polls in polling-fallback mode.
            timeout: Maximum seconds to wait.  Raises ``TimeoutError``
                on expiry.

        """
        api: InferenceApi = self._api  # type: ignore[assignment]

        def _sync_self(updated: InferenceJob) -> None:
            """Copy fields from *updated* into *self*."""
            for field_name, field_value in updated.model_dump().items():
                if field_value != MISSING_FIELD:
                    setattr(self, field_name, field_value)
            if on_status is not None:
                on_status(self)

        api.wait(self, on_status=_sync_self, poll_interval=poll_interval, timeout=timeout)
        return self