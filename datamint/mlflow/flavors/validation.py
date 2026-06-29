"""Local validation of a DatamintModel before deployment."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mlflow.models import ModelSignature
    from datamint.mlflow.flavors.model import BaseDatamintModel
    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities.resource import BaseResource

_LOGGER = logging.getLogger(__name__)

_TASK_ANNOTATION_TYPE: dict[str, str] = {
    'image_segmentation': 'segmentation',
    'instance_segmentation': 'segmentation',
    'video_segmentation': 'segmentation',
    'volume_segmentation': 'segmentation',
    'image_classification': 'category',
    'multilabel_image_classification': 'category',
    'volume_classification': 'category',
    'video_frame_classification': 'category',
    'object_detection': 'square',
}


@dataclass
class ValidationIssue:
    name: str
    passed: bool
    message: str
    severity: Literal['error', 'warning', 'info'] = 'error'


@dataclass
class ValidationReport:
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    signature: ModelSignature | None = None

    def __str__(self) -> str:
        lines = []
        for issue in self.issues:
            if issue.passed:
                marker = '[v]'
            elif issue.severity == 'error':
                marker = '[x]'
            else:
                marker = '[!]'
            lines.append(f'{marker} {issue.message}')

        warnings = sum(1 for i in self.issues if not i.passed and i.severity == 'warning')
        errors = sum(1 for i in self.issues if not i.passed and i.severity == 'error')

        if errors:
            summary = f'Failed with {errors} error{"s" if errors > 1 else ""}'
            if warnings:
                summary += f', {warnings} warning{"s" if warnings > 1 else ""}'
        elif warnings:
            summary = f'Passed with {warnings} warning{"s" if warnings > 1 else ""}'
        else:
            summary = 'Passed'

        lines.append(summary + '.')
        return '\n'.join(lines)


class ModelValidationError(Exception):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(str(report))


# --- checks ---

def _check_metadata(model: BaseDatamintModel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    task_type = getattr(model, 'task_type', None)
    if task_type is None:
        issues.append(ValidationIssue('task_type', False, 'task_type is not set', 'warning'))
    else:
        issues.append(ValidationIssue('task_type', True, f'task_type: {task_type}'))

    specs = getattr(model, 'annotation_specs', None)
    if not specs:
        issues.append(ValidationIssue('annotation_specs', False, 'annotation_specs is not set', 'warning'))
    else:
        issues.append(ValidationIssue('annotation_specs', True,
                                      f'annotation_specs: {", ".join(s.identifier for s in specs)}'))

    try:
        modes = model.get_supported_modes()
        if not modes:
            issues.append(ValidationIssue('supported_modes', False, 'no prediction modes registered', 'error'))
        else:
            issues.append(ValidationIssue('supported_modes', True, f'supported modes: {", ".join(modes)}'))
    except Exception as e:
        issues.append(ValidationIssue('supported_modes', False, f'get_supported_modes() raised: {e}', 'error'))

    return issues


def _check_inference(
    model: BaseDatamintModel,
    sample: list[BaseResource],
) -> tuple[list[ValidationIssue], ModelSignature | None]:
    issues: list[ValidationIssue] = []
    signature = None

    try:
        output = model.predict(sample)
        issues.append(ValidationIssue('predict', True, f'predict() OK on {len(sample)} sample(s)'))
    except Exception as e:
        issues.append(ValidationIssue('predict', False, f'predict() raised: {e}', 'error'))
        return issues, None

    if not isinstance(output, list):
        issues.append(ValidationIssue('output_type', False,
                                      f'output must be a list, got {type(output).__name__}', 'error'))
        return issues, None

    if len(output) != len(sample):
        issues.append(ValidationIssue('output_length', False,
                                      f'output length {len(output)} != input length {len(sample)}', 'error'))
    else:
        issues.append(ValidationIssue('output_length', True, f'output has {len(output)} element(s)'))
        empty = sum(1 for inner in output if not inner)
        if empty:
            issues.append(ValidationIssue('output_nonempty', False,
                                          f'{empty}/{len(output)} sample(s) returned no annotations', 'warning'))
        else:
            issues.append(ValidationIssue('output_nonempty', True, 'all predictions non-empty'))

    try:
        from datamint.mlflow.flavors.datamint_flavor import _process_signature
        signature = _process_signature(None, model)
        issues.append(ValidationIssue('signature', True, 'MLflow signature inferred'))
    except Exception as e:
        issues.append(ValidationIssue('signature', False, f'signature inference failed: {e}', 'warning'))

    if len(output) != len(sample):
        return issues, signature

    # annotation_specs consistency
    specs = getattr(model, 'annotation_specs', None)
    if specs:
        declared = {s.identifier for s in specs}
        required = {s.identifier for s in specs if s.required}
        found: set[str] = {
            ann.identifier
            for inner in output
            for ann in inner
            if getattr(ann, 'identifier', None)
        }
        unknown = found - declared
        if unknown:
            issues.append(ValidationIssue('spec_identifiers', False,
                                          f'identifiers not in annotation_specs: {", ".join(sorted(unknown))}',
                                          'error'))
        else:
            issues.append(ValidationIssue('spec_identifiers', True, 'output identifiers match annotation_specs'))

        missing = required - found
        if missing:
            issues.append(ValidationIssue('spec_required', False,
                                          f'required annotations not produced: {", ".join(sorted(missing))}',
                                          'warning'))
        else:
            issues.append(ValidationIssue('spec_required', True, 'all required annotation_specs satisfied'))

    # task_type consistency
    task_type = getattr(model, 'task_type', None)
    if task_type is not None:
        expected = _TASK_ANNOTATION_TYPE.get(str(task_type))
        if expected is not None:
            wrong = {
                str(getattr(ann, 'annotation_type', ''))
                for inner in output
                for ann in inner
                if str(getattr(ann, 'annotation_type', '')) not in ('', expected)
            }
            if wrong:
                issues.append(ValidationIssue('task_type_consistency', False,
                                              f'task_type {str(task_type)!r} expects {expected!r}, '
                                              f'got: {", ".join(sorted(wrong))}', 'error'))
            else:
                issues.append(ValidationIssue('task_type_consistency', True,
                                              f'annotation types consistent with task_type'))

    return issues, signature


# --- public API ---

def validate_model(
    model: BaseDatamintModel,
    dataset: DatamintBaseDataset | None = None,
    sample_input: list[BaseResource] | None = None,
    *,
    n_samples: int = 2,
) -> ValidationReport:
    """Validate a DatamintModel locally before deployment.

    Runs metadata checks and, when sample data is available, an inference
    smoke test with annotation_specs/task_type consistency checks.
    The inferred MLflow signature is attached to the report.
    """
    resolved: list[BaseResource] | None = sample_input
    if resolved is None and dataset is not None:
        resolved = list(dataset.resources[:n_samples])
        if not resolved:
            _LOGGER.warning('Dataset has no resources; inference checks will be skipped.')

    issues = _check_metadata(model)

    signature = None
    if resolved:
        tier2, signature = _check_inference(model, resolved)
        issues.extend(tier2)
    else:
        issues.append(ValidationIssue('inference', False,
                                      'no sample provided; inference checks skipped', 'warning'))

    passed = all(i.passed or i.severity != 'error' for i in issues)
    return ValidationReport(passed=passed, issues=issues, signature=signature)
