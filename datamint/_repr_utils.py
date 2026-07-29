"""Shared plain-text / Jupyter HTML repr rendering for entities, trainers, and datasets.

Any class that can produce a ``(label, value)`` field list gets a consistent
`print()` block and a consistent HTML card in Jupyter for free.
"""

# ---------------------------------------------------------------------------
# Jinja2 HTML template for the Jupyter card repr
# ---------------------------------------------------------------------------
_CARD_HTML_TEMPLATE = """\
<div style="max-width: 720px; margin: 10px 0; overflow: hidden; border-radius: 18px;
           border: 1px solid var(--vscode-panel-border, #d0d7de);
           background: var(--vscode-editor-background, #ffffff);
           color: var(--vscode-foreground, #1f2328);
           box-shadow: 0 10px 30px rgba(15, 23, 42, 0.10);">

  {# ---- Header ---- #}
  <div style="padding: 18px 20px;
             border-bottom: 1px solid var(--vscode-panel-border, #d0d7de);
             background: linear-gradient(135deg, rgba(59, 130, 246, 0.14), rgba(16, 185, 129, 0.08));">
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
               color: var(--vscode-descriptionForeground, #57606a);">{{ kind }}</div>
    <div style="display: flex; align-items: center; justify-content: space-between;
               gap: 12px; flex-wrap: wrap; margin-top: 8px;">
      <h4 style="margin: 0; font-size: 22px; font-weight: 700; color: inherit;">{{ name }}</h4>
    </div>
  </div>

  {# ---- Fields table ---- #}
  {%- if fields %}
  <div style="padding: 12px 20px 18px;">
    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
      {%- for label, value in fields %}
      <tr>
        <th style="padding: 10px 12px 10px 0; width: 30%; text-align: left; vertical-align: top;
                  font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
                  color: var(--vscode-descriptionForeground, #57606a); white-space: nowrap;">{{ label }}</th>
        <td style="padding: 10px 0; border-bottom: 1px solid var(--vscode-panel-border, #d0d7de);">
          <span style="display: inline-block; padding: 2px 8px; border-radius: 999px;
                      background: var(--vscode-textCodeBlock-background, #f6f8fa);
                      color: var(--vscode-textPreformat-foreground, var(--vscode-foreground, #1f2328));
                      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
                      font-size: 13px;"
          >{{ value }}</span>
        </td>
      </tr>
      {%- endfor %}
    </table>
  </div>
  {%- else %}
  <div style="padding: 18px 20px; font-size: 14px;
             color: var(--vscode-descriptionForeground, #57606a);">No non-empty fields to display.</div>
  {%- endif %}

</div>
"""

_card_template = None


def _get_card_template():
    """Lazily compile and cache the Jinja2 card template."""
    global _card_template
    if _card_template is None:
        from jinja2 import Environment
        _card_template = Environment(autoescape=True).from_string(_CARD_HTML_TEMPLATE)
    return _card_template


def render_text_block(header: str, fields: list[tuple[str, str]], empty_message: str = "(no non-empty fields)") -> str:
    """Plain-text ``Header\\n  Label: value`` block, used by ``__str__``/``__repr__``."""
    if not fields:
        return f"{header}\n  {empty_message}"
    lines = [header] + [f"  {label}: {value}" for label, value in fields]
    return "\n".join(lines)


def render_html_card(kind: str, name: str, fields: list[tuple[str, str]]) -> str:
    """Styled HTML card for Jupyter's ``_repr_html_`` display hook."""
    return _get_card_template().render(kind=kind, name=name, fields=fields)
