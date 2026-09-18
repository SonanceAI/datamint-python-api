"""API handler for validation study endpoints."""
import logging
from typing import TYPE_CHECKING, Any

import httpx

from datamint.entities.annotations.annotation_spec import AnnotationSpec
from datamint.entities.validation_study import ValidationStudy

from ..entity_base_api import ApiConfig, EntityBaseApi

if TYPE_CHECKING:
    from datamint.entities.project import Project

    from .models_api import ModelsApi
    from .projects_api import ProjectsApi


class ValidationStudiesApi(EntityBaseApi[ValidationStudy]):

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None,
                 models_api: 'ModelsApi | None' = None,
                 projects_api: 'ProjectsApi | None' = None) -> None:
        
        super().__init__(config, ValidationStudy, 'validation-studies', client)
        
        from .models_api import ModelsApi
        from .projects_api import ProjectsApi
        self._models_api = models_api or ModelsApi(config, client=client)
        self._projects_api = projects_api or ProjectsApi(config, client=client)

    @staticmethod
    def _specs_to_dicts(specs: list[Any]) -> list[dict[str, Any]]:
        return [s.asdict() if isinstance(s, AnnotationSpec) else s for s in specs]

    def create(self,
              project: 'str | Project',
              name: str,
              model_name: str,
              *,
              model_version: int | None = None,
              description: str | None = None,
              evaluate_ai_output: dict[str, Any] | None = None,
              provide_annotations: dict[str, Any] | None = None,
              resource_ids: list[str] | None = None,
              auto_fill_annotations: bool = True) -> ValidationStudy:
        """Create a draft validation study.

        Args:
            project: Project ID or Project instance the study belongs to.
            name: Study name.
            model_name: Registered model name. Cannot be changed after creation.
            model_version: Specific model version. Optional.
            description: Shown to readers as instructions. 
            evaluate_ai_output: Config for the "readers review the model's predictions"
                task. Dict with keys: ``enabled`` (bool), ``overall`` (bool, ask for an
                overall AI rating per file), ``indicate_errors`` (bool, let readers mark
                model errors on evaluated files), ``annotations`` (list of dicts or
                ``AnnotationSpec`` instances describing which model outputs to show).
                If ``enabled`` and ``annotations`` is omitted, they're fetched
                automatically from the deployed model (see ``auto_fill_annotations``).
            provide_annotations: Config for the "readers annotate from scratch" task.
                Dict with keys: ``enabled`` (bool), ``targets`` (list of dicts or
                ``AnnotationSpec`` instances the reader should fill in),
                ``segregate_count``.
            resource_ids: Optional initial set of resource IDs.
            auto_fill_annotations: If True (default) and ``evaluate_ai_output['enabled']``
                is True but no ``annotations`` were given, fetch them from
                ``api.models.get_deployed_model_info(project, model_name)``.

        Returns:
            The created draft ``ValidationStudy``.

        Raises:
            ValueError: If neither task is enabled, if ``segregate_count`` is given
                without both tasks enabled, or if annotation specs can't be
                auto-fetched and none were given explicitly.
        """
        evaluate_ai_output = dict(evaluate_ai_output) if evaluate_ai_output else {'enabled': False}
        provide_annotations = dict(provide_annotations) if provide_annotations else {'enabled': False}

        if not evaluate_ai_output.get('enabled') and not provide_annotations.get('enabled'):
            raise ValueError(
                "At least one of 'evaluate_ai_output' or 'provide_annotations' must be enabled."
            )

        both_enabled = bool(evaluate_ai_output.get('enabled') and provide_annotations.get('enabled'))
        if provide_annotations.get('segregate_count') is not None and not both_enabled:
            raise ValueError(
                "'segregate_count' is only meaningful when both 'evaluate_ai_output' and "
                "'provide_annotations' are enabled."
            )

        if (evaluate_ai_output.get('enabled')
                and not evaluate_ai_output.get('annotations')
                and auto_fill_annotations):
            info = self._models_api.get_deployed_model_info(project, model_name)
            specs = info.get('annotation_specs')
            if not specs:
                raise ValueError(
                    f"No annotation specs available for deployed model {model_name!r} "
                    f"(metadata_status={info.get('metadata_status')!r}). "
                    "Pass 'annotations' explicitly in 'evaluate_ai_output'."
                )
            evaluate_ai_output['annotations'] = specs

        if evaluate_ai_output.get('annotations'):
            evaluate_ai_output['annotations'] = self._specs_to_dicts(evaluate_ai_output['annotations'])
        if provide_annotations.get('targets'):
            provide_annotations['targets'] = self._specs_to_dicts(provide_annotations['targets'])

        payload: dict[str, Any] = {
            'name': name,
            'model_name': model_name,
            'reader_tasks': {
                'evaluate_ai_output': evaluate_ai_output,
                'provide_annotations': provide_annotations,
            },
        }
        if model_version is not None:
            payload['model_version'] = model_version
        if description is not None:
            payload['description'] = description
        if resource_ids is not None:
            payload['resource_ids'] = resource_ids

        project_id = self._entid(project)
        response = self._make_request('POST', f'/projects/{project_id}/validation-studies', json=payload)
        response_data = response.json()

        return self._init_entity_obj(**response_data)

    def set_resources(self, study: 'str | ValidationStudy', resource_ids: list[str]) -> None:
        """Replace the file set for a draft validation study. Only works while it's a draft.

        Args:
            study: The validation study ID or instance.
            resource_ids: The full new set of resource IDs.
        """
        self._make_entity_request('PUT', study, 'resources', json={'resource_ids': resource_ids})

    def add_readers(self, study: 'str | ValidationStudy', readers: list[dict[str, str]]) -> None:
        """Add readers to a validation study.

        Args:
            study: The validation study ID or instance.
            readers: List of dicts with keys ``email`` (required), ``firstname``, ``lastname``.
        """
        self._make_entity_request('POST', study, 'readers', json={'readers': readers})
