"""Model evaluation against a fixed dataset: predict, score, log to MLflow. """
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypeAlias

import mlflow

from datamint.api.endpoints.model_types import Model, ModelVersion
from datamint.entities.annotations import AnnotationType
from datamint.mlflow.flavors.model import BaseDatamintModel
from datamint.mlflow.flavors.task_type import TaskType
from datamint.mlflow.tracking.fluent import (
    _reset_active_project,
    get_active_project_id,
    set_project,
)

from .metrics import (
    ClassificationScores,
    DetectionScores,
    SegmentationScores,
    compute_classification_scores,
    compute_detection_scores,
    compute_segmentation_scores,
)

if TYPE_CHECKING:
    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities.annotations.annotation import Annotation

_LOGGER = logging.getLogger(__name__)

# task_type -> (ground-truth/prediction filter, scoring function)
_TASK_TYPE_HANDLERS: dict[TaskType, tuple[Callable[['Annotation'], bool], Callable]] = {
    TaskType.IMAGE_SEGMENTATION: (lambda a: a.is_segmentation(), compute_segmentation_scores),
    TaskType.VOLUME_SEGMENTATION: (lambda a: a.is_segmentation(), compute_segmentation_scores),
    TaskType.IMAGE_CLASSIFICATION: (lambda a: a.is_category(), compute_classification_scores),
    TaskType.OBJECT_DETECTION: (lambda a: a.annotation_type == AnnotationType.SQUARE.value, compute_detection_scores),
}
_DEFAULT_EXPERIMENT_NAME = 'Evaluation'

ModelEntry: TypeAlias = str | Model | ModelVersion | BaseDatamintModel | tuple[Any, dict[str, Any]]


@dataclass
class EvaluationResult:
    """Predictions + scores for one model from a single ``.evaluate()`` call """

    model_name: str
    predictions: 'list[list[Annotation]]'
    scores: SegmentationScores | ClassificationScores | DetectionScores
    hyperparameters: dict[str, Any]
    mlflow_run_id: str | None


@dataclass
class _ResolvedModel:
    model_name: str
    dm_model: BaseDatamintModel | None
    version: ModelVersion | None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """Disambiguated identity for the results dict key and MLflow run name. """
        if self.version is not None:
            return f"{self.model_name}_v{self.version.version}"
        return self.model_name


class Evaluator:
    """Evaluate one or more models against a fixed dataset.

    The dataset is set once at construction and reused by every
    ``.evaluate()`` call, so a hyperparameter sweep or model comparison
    doesn't need to re-pass it each time::

        evaluator = Evaluator(dataset=my_dataset)
        results = evaluator.evaluate(models=['MyModel1', 'MyModel2'])
    """

    def __init__(self, dataset: 'DatamintBaseDataset') -> None:
        self.dataset = dataset

    def evaluate(
        self,
        models: ModelEntry | Sequence[ModelEntry],
        *,
        prefer_deployed: bool = False,
        log_to_mlflow: bool = True,
        experiment_name: str = _DEFAULT_EXPERIMENT_NAME,
        save_results: bool = False,
    ) -> dict[str, EvaluationResult]:
        """Predict and score every entry in ``models`` against the fixed dataset.

        Args:
            models: A single entry, or a sequence of entries. Each entry is a
                registered model name, a ``Model``, a ``ModelVersion``, a bare
                ``BaseDatamintModel`` instance, or a ``(model, config_dict)``
                pair. ``config_dict`` is passed as ``predict()`` params for
                local models and merged into the logged hyperparameters (this
                is the only way to attach hyperparameters for a bare,
                never-registered model instance). A single string/``Model``/
                ``ModelVersion``/``BaseDatamintModel`` (e.g.
                ``models="unetpp_r34"``) is accepted directly.
            prefer_deployed: If a model is both registered and deployed, use
                the deployed/remote path instead of loading it locally.
                Defaults to False.
            log_to_mlflow: If ``True``, log hyperparameters and scores
                (Dice/IoU for segmentation, accuracy/F1 for classification,
                mAP for detection) to MLflow.
            experiment_name: MLflow experiment to log into. Defaults to
                ``'Evaluation'``. If that name is blocked, falls back to
                ``f'{experiment_name}_<timestamp>'`` instead of raising.
            save_results: If ``True``, upload each model's predictions back to
                Datamint as annotations on their resources. 

        Returns:
            Dict keyed by model name.
        """
        if isinstance(models, (str, Model, ModelVersion, BaseDatamintModel)):
            models = [models]

        api = self.dataset._api
        resources = list(self.dataset.resources)
        resource_ids = [r.id for r in resources]
        resource_annotations = self.dataset.resource_annotations

        results: dict[str, EvaluationResult] = {}
        project = getattr(self.dataset, 'project', None)
        previous_project_id = get_active_project_id()

        try:
            if project is not None:
                set_project(project)

            resolved_models = [self._resolve_entry(entry, api, prefer_deployed) for entry in models]
            evaluation_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

            if log_to_mlflow:
                self._set_experiment(experiment_name)

            for resolved in resolved_models:
                try:
                    task_type = self._task_type(resolved)
                    handler = _TASK_TYPE_HANDLERS.get(task_type)
                    if handler is None:
                        raise NotImplementedError(
                            "evaluate() currently only supports segmentation, "
                            "single-label classification and object detection task types "
                            f"({', '.join(t.name for t in _TASK_TYPE_HANDLERS)}); got "
                            f"{task_type!r} for {resolved.model_name!r}."
                        )
                    annotation_filter, compute_scores = handler

                    ground_truths = [
                        [a for a in anns if annotation_filter(a)] for anns in resource_annotations
                    ]
                    predictions = self._predict(resolved, resources, api, save_results)
                    filtered_preds = [[a for a in anns if annotation_filter(a)] for anns in predictions]
                    scores = compute_scores(resource_ids, ground_truths, filtered_preds)

                    hyperparameters: dict[str, Any] = {}
                    if resolved.version is not None:
                        hyperparameters.update(resolved.version.get_hyperparameters())
                    hyperparameters.update(resolved.config)

                    run_id = (
                        self._log_to_mlflow(resolved, scores, hyperparameters, evaluation_id)
                        if log_to_mlflow else None
                    )

                    results[resolved.display_name] = EvaluationResult(
                        model_name=resolved.display_name,
                        predictions=predictions,
                        scores=scores,
                        hyperparameters=hyperparameters,
                        mlflow_run_id=run_id,
                    )
                except Exception:
                    _LOGGER.exception(
                        "Evaluation failed for %r; skipping it and continuing with the "
                        "remaining models.", resolved.display_name,
                    )
        finally:
            if project is not None:
                if previous_project_id is not None:
                    set_project(previous_project_id)
                else:
                    _reset_active_project()

        return results

    @staticmethod
    def _set_experiment(experiment_name: str) -> None:
        try:
            mlflow.set_experiment(experiment_name)
        except mlflow.exceptions.MlflowException as e:
            error_code = getattr(e, 'error_code', None)
            if error_code != 'INVALID_PARAMETER_VALUE' or 'deleted experiment' not in str(e).lower():
                raise
            fallback_name = f"{experiment_name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            _LOGGER.warning(
                "Experiment %r is soft-deleted and can't be reused; logging to "
                "%r instead. Restore or permanently delete it to reuse the name.",
                experiment_name, fallback_name,
            )
            mlflow.set_experiment(fallback_name)

    def _resolve_entry(self, entry: ModelEntry, api, prefer_deployed: bool) -> _ResolvedModel:
        config: dict[str, Any] = {}
        if isinstance(entry, tuple):
            entry, config = entry

        if isinstance(entry, BaseDatamintModel):
            return _ResolvedModel(
                model_name=type(entry).__name__, dm_model=entry, version=None, config=config
            )

        if isinstance(entry, ModelVersion):
            version = entry
            model_name = version.name
        elif isinstance(entry, Model):
            model_name = entry.name
            version = entry.get_latest_version()
            if version is None:
                raise ValueError(f"Model {model_name!r} has no versions.")
        elif isinstance(entry, str):
            model = api.models.get_by_name(entry)
            if model is None:
                raise ValueError(f"No registered model named {entry!r}.")
            model_name = entry
            version = model.get_latest_version()
            if version is None:
                raise ValueError(f"Model {entry!r} has no versions.")
        else:
            raise TypeError(f"Unsupported models[] entry: {entry!r}")

        use_deployed = prefer_deployed and version.is_deployed()
        dm_model = None if use_deployed else version.load_model()
        return _ResolvedModel(model_name=model_name, dm_model=dm_model, version=version, config=config)

    @staticmethod
    def _task_type(resolved: _ResolvedModel) -> TaskType | None:
        if resolved.dm_model is not None:
            return resolved.dm_model.task_type
        task_type_str = resolved.version.get_task_type() if resolved.version else None
        return TaskType(task_type_str) if task_type_str else None

    def _predict(
        self, resolved: _ResolvedModel, resources, api, save_results: bool
    ) -> 'list[list[Annotation]]':
        if resolved.dm_model is not None:
            params = dict(resolved.config)
            if save_results:
                params.setdefault('log_predictions', True)
                params.setdefault('model_name', resolved.model_name)
            return resolved.dm_model.predict(resources, params=params or None)

        version = resolved.version
        assert version is not None
        predictions: list[list[Annotation]] = []
        for resource in resources:
            job = api.inference.predict_image(
                model_name=version.name,
                model_version=int(version.version),
                resource_id=resource.id,
                save_results=save_results,
                params=resolved.config or None,
            )
            job.wait()
            preds = job.predictions
            predictions.append(preds[0] if preds else [])
        return predictions

    @staticmethod
    def _log_to_mlflow(
        resolved: _ResolvedModel,
        scores: SegmentationScores | ClassificationScores | DetectionScores,
        hyperparameters: dict[str, Any],
        evaluation_id: str,
    ) -> str:
        with mlflow.start_run(run_name=resolved.display_name) as child_run:
            mlflow.set_tag('evaluation_id', evaluation_id)
            if resolved.version is not None:
                mlflow.set_tag('model_name', resolved.model_name)
                mlflow.set_tag('model_version', resolved.version.version)
            if hyperparameters:
                mlflow.log_params(hyperparameters)
            if scores.dataset:
                mlflow.log_metrics(scores.dataset)
            for class_name, class_scores in scores.per_class.items():
                mlflow.log_metrics({
                    f'{metric}_{class_name}': value for metric, value in class_scores.items()
                })
            return child_run.info.run_id
