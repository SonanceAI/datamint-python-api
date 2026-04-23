"""
DatamintBaseDataset - Abstract base class for all Datamint datasets.

Provides the PyTorch Dataset interface with transform support and annotation
filtering, while delegating data management to DatamintProjectManager.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence, Callable, Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from datamint.exceptions import DatamintException
from .annotation_processor import AnnotationProcessor, MergeStrategy
from datamint.entities.annotations.annotation_spec import AnnotationSpec, CategoryAnnotationSpec
from datamint.entities.annotations import AnnotationType

if TYPE_CHECKING:
    from datamint.entities import Resource, Project, Annotation
    from albumentations import BaseCompose
    from datamint.mlflow.data import DatamintMLflowDataset

_LOGGER = logging.getLogger(__name__)


class DatamintDatasetException(DatamintException):
    """Exception raised for dataset errors."""
    pass


class DatamintBaseDataset(ABC, torch.utils.data.Dataset):
    """Abstract base class for Datamint datasets.

    This class provides the PyTorch Dataset interface with:
    - Transform hooks (albumentations)
    - Annotation filtering
    - Data loading utilities

    Subclasses must implement `_get_raw_item()` to define how data is loaded.

    Args:
        project: Project name, Project object, or None. Mutually exclusive with resources.
        resources: List of Resource objects/IDs, or None. Mutually exclusive with project.
        auto_update: If True, sync with server on init.
        api_key: API key for authentication.
        server_url: Datamint server URL.
        all_annotations: If True, include unpublished annotations.
        return_metainfo: If True, include metadata in output.
        return_segmentations: If True, process and return segmentations.
        return_as_semantic_segmentation: If True, convert to semantic format.
        semantic_seg_merge_strategy: Strategy for merging multi-annotator segs.
        alb_transform: Albumentations transform.
        include_unannotated: If True, include resources without annotations.
        include_annotators: Whitelist of annotators.
        exclude_annotators: Blacklist of annotators.
        include_segmentation_names: Whitelist of segmentation labels.
        exclude_segmentation_names: Blacklist of segmentation labels.
        include_image_label_names: Whitelist of image labels.
        exclude_image_label_names: Blacklist of image labels.
        include_frame_label_names: Whitelist of frame labels.
        exclude_frame_label_names: Blacklist of frame labels.
        allow_external_annotations: If True, allow and automatically include annotation
            labels that are not part of the project's official schema (e.g., labels
            from other projects or legacy annotations). If False, these annotations
            will be filtered out.
    """

    resources: Sequence['Resource']
    resource_annotations: list[Sequence['Annotation']]
    project: 'Project | None'

    def __init__(
        self,
        project: 'str | Project | None' = None,
        resources: 'Sequence[Resource] | None' = None,
        auto_update: bool = True,
        api_key: str | None = None,
        server_url: str | None = None,
        # all_annotations: bool = False,
        return_metainfo: bool = True,
        return_segmentations: bool = True,
        return_as_semantic_segmentation: bool = False,
        semantic_seg_merge_strategy: MergeStrategy | None = None,
        alb_transform: 'Callable | BaseCompose | None' = None,
        include_unannotated: bool = True,
        include_annotators: list[str] | None = None,
        exclude_annotators: list[str] | None = None,
        include_segmentation_names: list[str] | None = None,
        exclude_segmentation_names: list[str] | None = None,
        include_image_label_names: list[str] | None = None,
        exclude_image_label_names: list[str] | None = None,
        include_frame_label_names: list[str] | None = None,
        exclude_frame_label_names: list[str] | None = None,
        allow_external_annotations: bool = False,
        image_labels_merge_strategy: MergeStrategy | None = None,
        image_categories_merge_strategy: MergeStrategy | None = None,
    ):
        # Validate merge strategy values
        _valid_strategies = ('union', 'intersection', 'mode', None)
        if image_labels_merge_strategy not in _valid_strategies:
            raise ValueError(
                f"image_labels_merge_strategy must be one of {_valid_strategies[:-1]!r}, "
                f"got {image_labels_merge_strategy!r}"
            )
        if image_categories_merge_strategy not in _valid_strategies:
            raise ValueError(
                f"image_categories_merge_strategy must be one of {_valid_strategies[:-1]!r}, "
                f"got {image_categories_merge_strategy!r}"
            )

        # Validate mutually exclusive parameters
        if project is not None and resources is not None:
            raise DatamintDatasetException(
                "Cannot specify both 'project' and 'resources'. Choose one."
            )

        if project is None and resources is None:
            raise DatamintDatasetException(
                "Must provide either 'project' or 'resources'."
            )

        # Validate filtering parameters
        self._validate_filter_params(
            include_annotators, exclude_annotators,
            include_segmentation_names, exclude_segmentation_names,
            include_image_label_names, exclude_image_label_names,
            include_frame_label_names, exclude_frame_label_names
        )

        # Validate segmentation parameters
        if not return_segmentations and return_as_semantic_segmentation:
            raise ValueError("Cannot return semantic segmentation without returning segmentations.")
        if semantic_seg_merge_strategy and not return_as_semantic_segmentation:
            raise ValueError("semantic_seg_merge_strategy requires return_as_semantic_segmentation=True")

        # Store API configuration for (possibly deferred) initialization
        self._server_url = server_url
        self._api_key = api_key
        self._auto_update = auto_update
        self._init_project = project
        self._init_resources = resources

        # Store configuration
        self.return_metainfo = return_metainfo
        self.return_segmentations = return_segmentations
        self.return_as_semantic_segmentation = return_as_semantic_segmentation
        self.semantic_seg_merge_strategy: MergeStrategy | None = semantic_seg_merge_strategy
        self.include_unannotated = include_unannotated

        # Transforms
        self.set_transform(alb_transform)

        # Filtering
        self.include_annotators = include_annotators
        self.exclude_annotators = exclude_annotators
        self.include_segmentation_names = include_segmentation_names
        self.exclude_segmentation_names = exclude_segmentation_names
        self.include_image_label_names = include_image_label_names
        self.exclude_image_label_names = exclude_image_label_names
        self.include_frame_label_names = include_frame_label_names
        self.exclude_frame_label_names = exclude_frame_label_names
        self.allow_external_annotations = allow_external_annotations
        self.image_labels_merge_strategy: MergeStrategy | None = image_labels_merge_strategy
        self.image_categories_merge_strategy: MergeStrategy | None = image_categories_merge_strategy

        # Internal state
        self._logged_uint16_conversion = False
        self._is_prepared = False
        self.split_name: str | None = None
        self.split_source: str | None = None
        self.split_as_of_timestamp: str | None = None

    @staticmethod
    def _utc_now_isoformat() -> str:
        """Return the current UTC timestamp in ISO-8601 format."""
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only invoked when normal attribute lookup fails —
        # i.e. for attributes populated by _prepare() (resources, project,
        # label sets, annotation_processor, …). Guard against recursion for
        # attributes not yet written during __init__ itself.
        if self.__dict__.get('_is_prepared') is not False:
            raise AttributeError(name)
        self._prepare()
        return object.__getattribute__(self, name)

    def _prepare(self) -> None:
        """Fetch data from the API and set up the dataset.

        Called automatically on first access of any attribute that requires
        server data (e.g. ``resources``, ``project``, label sets).
        Idempotent — safe to call multiple times.
        """
        if self._is_prepared:
            return

        from datamint import Api

        # Initialize API
        self._api = Api(
            server_url=self._server_url,
            api_key=self._api_key,
            check_connection=self._auto_update,
        )

        # Initialize from project or resources
        if self._init_resources is not None:
            self.resources = self._initialize_from_resources(self._init_resources)
            self.project = None
        else:
            self.project, self.resources = self._initialize_from_project(self._init_project)  # type: ignore

        if len(self.resources) == 0:
            _LOGGER.warning("Initialized dataset with no resources.")
        # Fetch annotations
        self.resource_annotations = list(self._api.annotations.get_list(
            resource=self.resources,
            group_by_resource=True,
        ))

        # Setup dataset (labels, annotation processor, filters)
        self._setup_dataset()

        self._is_prepared = True

        # Clean up temporary attributes used only for deferred initialisation
        del self._server_url, self._api_key, self._auto_update
        del self._init_project, self._init_resources


    def prefetch(self, *, include_annotations: bool = False) -> None:
        """Download and cache dataset files eagerly.

        Ensures that resource file bytes are present in the local cache before
        training begins, so ``__getitem__`` calls are served from disk rather
        than triggering on-demand network requests. When ``include_annotations``
        is enabled, segmentation annotation payloads are cached too so
        DataLoader workers do not need to fetch them from the API.

        Calls ``_prepare()`` implicitly if the dataset has not been initialised yet.

        Args:
            include_annotations: Whether to also prefetch segmentation
                annotation payloads.
        """
        _LOGGER.info(f"Prefetching {len(self.resources)} resource(s)...")
        requires_download = any(not r.is_cached() for r in self.resources)
        iterator = iter(self.resources)
        if requires_download:
            from tqdm.auto import tqdm
            _LOGGER.warning(
                "Some resources are not cached locally and will be downloaded during slicing. "
                "This may take time and bandwidth, especially for large volumes. "
                "Consider pre-caching resources if this is an issue.")
            iterator = tqdm(iterator, total=len(self.resources),
                             desc="Prefetching resources", unit="resource")
        for resource in iterator:
            try:
                resource.fetch_file_data(auto_convert=False, use_cache=True)
            except Exception as e:
                _LOGGER.warning(f"Failed to prefetch resource '{resource.id}': {e}")

        if include_annotations:
            segmentation_annotations = []
            seen_annotation_keys: set[str] = set()

            iterator = iter(self.resource_annotations)
            requires_download = any(not ann.is_cached() for anns in self.resource_annotations for ann in anns if ann.is_segmentation())
            if requires_download:
                from tqdm.auto import tqdm
                _LOGGER.warning(
                    "Some segmentation annotations are not cached locally and will be downloaded. "
                    "This may take time and bandwidth, especially for large volumes. "
                    "Consider pre-caching annotations if this is an issue.")
                iterator = tqdm(iterator, total=len(self.resource_annotations),
                                 desc="Prefetching annotations", unit="resource")
            for annotations in iterator:
                for annotation in annotations:
                    annotation_id = getattr(annotation, 'id', None)
                    annotation_key = annotation_id or f'object:{id(annotation)}'
                    if annotation_key in seen_annotation_keys:
                        continue

                    is_segmentation = getattr(annotation, 'annotation_type', None) == 'segmentation'
                    if not is_segmentation:
                        is_segmentation_method = getattr(annotation, 'is_segmentation', None)
                        if callable(is_segmentation_method):
                            try:
                                is_segmentation = bool(is_segmentation_method())
                            except Exception:
                                is_segmentation = False

                    if not is_segmentation:
                        continue

                    seen_annotation_keys.add(annotation_key)
                    segmentation_annotations.append(annotation)

            for annotation in segmentation_annotations:
                try:
                    annotation.fetch_file_data(auto_convert=False, use_cache=True)
                except Exception as e:
                    _LOGGER.warning(f"Failed to prefetch annotation '{getattr(annotation, 'id', None)}': {e}")

        _LOGGER.info("Prefetch complete.")

    # def __getstate__(self) -> object:
    #     # print the size in MB
    #     print(f'>>>{len(str(self.__dict__))/1024/1024:.2f} MB')

    #     return super().__getstate__()

    def __setstate__(self, state: dict) -> None:
        vars(self).update(state)
        if self._is_prepared:
            self._reinit_api()

    def _reinit_api(self) -> None:
        """Re-inject sub-API handles into entities after unpickling in a worker.

        ``self._api`` is already a fresh ``Api`` instance (reconstructed by
        ``Api.__setstate__``); we only need to wire its per-resource/annotation
        sub-APIs back into the individual entity objects.
        """
        resources_api = self._api.resources
        annotations_api = self._api.annotations
        projects_api = self._api.projects
        for res in self.resources:
            res._api = resources_api
        if self.project is not None:
            self.project._api = projects_api
        for ann_list in self.resource_annotations:
            for ann in ann_list:
                ann._api = annotations_api

    def set_transform(self, alb_transform: 'BaseCompose | None' = None) -> None:
        """Set transforms after initialization."""
        self.alb_transform = alb_transform

    def add_transform(self, alb_transform: 'BaseCompose') -> None:
        import albumentations as A
        """Add an albumentations transform, composing with existing one if present."""
        if self.alb_transform is not None:
            self.alb_transform = A.Compose([self.alb_transform, alb_transform])
        else:
            self.alb_transform = alb_transform

    def _extract_image_labels(
        self,
        annotations: Sequence['Annotation'],
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Extract image-level label annotations.

        When ``image_labels_merge_strategy`` is ``None`` (default), returns a dict
        mapping each annotator id to its binary label tensor of shape ``(num_labels,)``.
        When a strategy is set, returns a single merged tensor of the same shape.

        Args:
            annotations: All annotations for the item.

        Returns:
            Dict[annotator_id, Tensor] or merged Tensor depending on
            :attr:`image_labels_merge_strategy`.
        """
        label_annotations = AnnotationProcessor.filter_annotations(
            annotations, type='label', scope='image'
        )
        labels_dict = self.annotation_processor.convert_image_labels(label_annotations)
        if self.image_labels_merge_strategy is None:
            return labels_dict
        return self.annotation_processor.merge_image_labels(
            labels_dict, self.image_labels_merge_strategy
        )

    def _extract_image_categories(
        self,
        annotations: Sequence['Annotation'],
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Extract image-level category annotations.

        When ``image_categories_merge_strategy`` is ``None`` (default), returns a dict
        mapping each annotator id to a scalar long tensor with the class index.
        When a strategy is set:
        - ``'mode'``: scalar long tensor (majority class index, -1 if no annotations).
        - ``'union'``/``'intersection'``: multi-hot int tensor of shape ``(num_categories,)``.

        Args:
            annotations: All annotations for the item.

        Returns:
            Dict[annotator_id, Tensor] or merged Tensor depending on
            :attr:`image_categories_merge_strategy`.
        """
        category_annotations = AnnotationProcessor.filter_annotations(
            annotations, type='category', scope='image'
        )
        categories_dict = self.annotation_processor.convert_image_categories(category_annotations)
        if self.image_categories_merge_strategy is None:
            return categories_dict
        return AnnotationProcessor.merge_image_categories(
            categories_dict,
            self.image_categories_merge_strategy,
            num_categories=len(self.image_categories_set),
        )

    def _validate_filter_params(
        self,
        include_annotators, exclude_annotators,
        include_segmentation_names, exclude_segmentation_names,
        include_image_label_names, exclude_image_label_names,
        include_frame_label_names, exclude_frame_label_names
    ) -> None:
        """Validate mutually exclusive filter parameters."""
        pairs = [
            (include_annotators, exclude_annotators, "annotators"),
            (include_segmentation_names, exclude_segmentation_names, "segmentation_names"),
            (include_image_label_names, exclude_image_label_names, "image_label_names"),
            (include_frame_label_names, exclude_frame_label_names, "frame_label_names"),
        ]
        for include_param, exclude_param, name in pairs:
            if include_param is not None and exclude_param is not None:
                raise DatamintDatasetException(f"Cannot specify both include_{name} and exclude_{name}.")

    def _initialize_from_project(
        self,
        project: 'str | Project',
    ) -> tuple['Project', list['Resource']]:
        """Initialize dataset from a project (name or object)."""

        # Handle Project object vs string
        if isinstance(project, str):
            project = self._api.projects.get_by_name(project)
            if project is None:
                raise DatamintDatasetException(f"Project '{project}' not found.")
        else:
            # Attach API to project if not already set
            if not hasattr(project, '_api') or project._api is None:
                project._api = self._api.projects

        # Fetch resources
        resources = list(project.fetch_resources())
        return project, resources

    def _initialize_from_resources(
        self,
        resources: Sequence['Resource'],
    ) -> Sequence['Resource']:
        """Initialize dataset from a list of resources."""
        for res in resources:
            if not hasattr(res, '_api') or res._api is None:
                res._api = self._api.resources

        return resources

    def _setup_dataset(self) -> None:
        """Setup dataset after initialization."""
        if not self.resources:
            _LOGGER.warning("No resources found in the dataset.")

        # Setup labels
        self._setup_labels()

        # Setup annotation processor
        self._setup_annotation_processor()

        # Apply annotation filters
        self._apply_annotation_filters()

        # Filter unannotated if needed
        orig_num_resources = len(self.resources)

        if not self.include_unannotated:
            self._filter_unannotated()
        
        if len(self.resources) == 0 and orig_num_resources > 0:
            _LOGGER.warning("All resources have been filtered out.")

    def _setup_labels(self) -> None:
        """Setup label sets and mappings."""

        if self.project is not None:
            worklist_schema = self._api.annotationsets._get_by_id(self.project.worklist_id)
            annotations_specs: Sequence['AnnotationSpec'] = worklist_schema['annotations']
            frame_annotations_specs = [annspec for annspec in annotations_specs
                                       if annspec.scope == 'frame']
            image_annotations_specs = [annspec for annspec in annotations_specs
                                       if annspec.scope == 'image']
            self.frame_lsets, self.frame_lcodes = self._process_annotation_specs(frame_annotations_specs)
            self.image_lsets, self.image_lcodes = self._process_annotation_specs(image_annotations_specs)

            # Segmentation labels
            self.seglabel_list, self.seglabel2code = self._process_segmentation_group(
                worklist_schema['segmentation_group']
            )

            if self.allow_external_annotations:
                self._augment_labels_from_annotations()
        else:
            _LOGGER.info("No project provided; inferring labels from annotations.")
            self.frame_lsets, self.frame_lcodes = self._infer_labels_set(framed=True)
            self.image_lsets, self.image_lcodes = self._infer_labels_set(framed=False)
            self.seglabel_list, self.seglabel2code = self._infer_segmentation_group()

    def _augment_labels_from_annotations(self) -> None:
        """Augment project-defined label sets with identifiers found in actual annotations.

        Scans resource annotations for identifiers not present in the project's
        annotations_specs and adds them to the corresponding label/segmentation mappings.
        """
        inferred_frame_lsets, inferred_frame_lcodes = self._infer_labels_set(framed=True)
        inferred_image_lsets, inferred_image_lcodes = self._infer_labels_set(framed=False)
        inferred_seglabel_list, _ = self._infer_segmentation_group()

        # Augment frame labels
        for kind in ('multilabel', 'multiclass'):
            existing = set(self.frame_lsets[kind])
            new_labels = sorted([label for label in inferred_frame_lsets[kind] if label not in existing])
            for label in new_labels:
                _LOGGER.info(f"Allowing external frame label '{label}' not in project specs.")
                self.frame_lsets[kind].append(label)
            self.frame_lcodes[kind] = self.__build_label_codemap(self.frame_lsets[kind])

        # Augment image labels
        for kind in ('multilabel', 'multiclass'):
            existing = set(self.image_lsets[kind])
            new_labels = sorted([label for label in inferred_image_lsets[kind] if label not in existing])
            for label in new_labels:
                _LOGGER.info(f"Allowing external image label '{label}' not in project specs.")
                self.image_lsets[kind].append(label)
            self.image_lcodes[kind] = self.__build_label_codemap(self.image_lsets[kind])

        # Augment segmentation labels
        existing_segs = set(self.seglabel_list)
        new_segs = sorted([label for label in inferred_seglabel_list if label not in existing_segs])
        for label in new_segs:
            _LOGGER.info(f"Allowing external segmentation label '{label}' not in project specs.")
            self.seglabel_list.append(label)
            self.seglabel2code[label] = len(self.seglabel_list)  # 1-based code

    def _setup_annotation_processor(self) -> None:
        """Initialize the annotation processor.

        Calls _create_annotation_processor() which subclasses can override
        to return the appropriate processor type.
        """
        self.annotation_processor = AnnotationProcessor(
            seglabel2code=self.seglabel2code,
            image_labels_set=self.image_labels_set,
            image_lcodes=self.image_lcodes,
            allow_external_annotations=self.allow_external_annotations,
        )

    def _apply_annotation_filters(self) -> None:
        """Apply annotation filters to all resources."""
        for i in range(len(self.resources)):
            anns = self.resource_annotations[i]
            filtered = self._filter_annotations(anns)
            self.resource_annotations[i] = filtered

    @abstractmethod
    def _get_raw_item(self, index: int) -> dict[str, Any]:
        """Load raw data for the given index.

        Must return dict with at least:
        - 'image': Tensor
        - 'metainfo': dict
        - 'annotations': list[Annotation]
        """
        pass

    def _filter_unannotated(self) -> None:
        """Filter out indices without annotations."""
        filtered_resources = []
        filtered_annotations = []

        for resource, annotations in zip(self.resources, self.resource_annotations):
            if annotations:
                filtered_resources.append(resource)
                filtered_annotations.append(annotations)

        self.resources = filtered_resources
        self.resource_annotations = filtered_annotations

    def _filter_annotations(self, annotations: Sequence['Annotation']) -> list['Annotation']:
        """Filter annotations based on include/exclude settings."""
        return [ann for ann in annotations if self._should_include_annotation(ann)]

    def _should_include_annotation(self, ann: 'Annotation') -> bool:
        """Check if annotation should be included."""
        # Check annotator
        annotator = ann.created_by
        if annotator is not None and not self._should_include_annotator(annotator):
            return False

        # Check by annotation type
        if ann.is_segmentation():
            return self._should_include_segmentation(ann.identifier)
        elif ann.is_label():
            if ann.frame_index is None:  # image-level
                return self._should_include_image_label(ann.identifier)
            else:  # frame-level
                return self._should_include_frame_label(ann.identifier)
        elif ann.is_category():
            if not self.allow_external_annotations:
                lsets = self.image_lsets if ann.frame_index is None else self.frame_lsets
                valid_identifiers = {ident for ident, _ in lsets.get('multiclass', [])}
                if ann.identifier not in valid_identifiers:
                    return False

        return True

    def _should_include_annotator(self, annotator_id: str) -> bool:
        if self.include_annotators is not None:
            return annotator_id in self.include_annotators
        if self.exclude_annotators is not None:
            return annotator_id not in self.exclude_annotators
        return True

    def _should_include_segmentation(self, name: str) -> bool:
        if not self.allow_external_annotations and name not in self.segmentation_labels_set:
            return False
        if self.include_segmentation_names is not None:
            return name in self.include_segmentation_names
        if self.exclude_segmentation_names is not None:
            return name not in self.exclude_segmentation_names
        return True

    def _should_include_image_label(self, name: str) -> bool:
        if not self.allow_external_annotations and name not in self.image_labels_set:
            return False
        if self.include_image_label_names is not None:
            return name in self.include_image_label_names
        if self.exclude_image_label_names is not None:
            return name not in self.exclude_image_label_names
        return True

    def _should_include_frame_label(self, name: str) -> bool:
        if not self.allow_external_annotations and name not in self.frame_labels_set:
            return False
        if self.include_frame_label_names is not None:
            return name in self.include_frame_label_names
        if self.exclude_frame_label_names is not None:
            return name not in self.exclude_frame_label_names
        return True

    def _process_annotation_specs(self,
                                  annotations_specs: Sequence['AnnotationSpec']
                                  ) -> tuple[dict[str, list], dict[str, dict[str, int]]]:
        multilabel_list: list[str] = []
        multiclass_list: list[tuple[str, str]] = []

        for annspec in annotations_specs:
            if annspec.type == AnnotationType.LABEL:
                multilabel_list.append(annspec.identifier)
            elif isinstance(annspec, CategoryAnnotationSpec):
                for val in annspec.values:
                    multiclass_list.append((annspec.identifier, val))

        sets = {
            'multilabel': multilabel_list,
            'multiclass': multiclass_list
        }
        codes = {
            'multilabel': self.__build_label_codemap(multilabel_list),
            'multiclass': self.__build_label_codemap(multiclass_list)
        }

        return sets, codes

    @staticmethod
    def __build_label_codemap(labels: Sequence) -> dict[str, int]:
        """Build label to code mapping."""
        return {label: idx for idx, label in enumerate(labels)}

    def _infer_labels_set(self, framed: bool) -> tuple[dict[str, list], dict[str, dict[str, int]]]:
        """Get label sets and codes."""
        scope = 'frame' if framed else 'image'
        multilabel_set: set[str] = set()
        multiclass_set: set[tuple[str, str]] = set()

        for annotations in self.resource_annotations:
            for ann in annotations:
                ann_scope = 'image' if ann.frame_index is None else 'frame'
                if ann_scope != scope:
                    continue

                if ann.is_label():
                    multilabel_set.add(ann.identifier)
                elif ann.is_category():
                    multiclass_set.add((ann.identifier, ann.value))

        multilabel_list = sorted(multilabel_set)
        multiclass_list = sorted(multiclass_set)

        sets = {
            'multilabel': multilabel_list,
            'multiclass': multiclass_list
        }
        codes = {
            'multilabel': self.__build_label_codemap(multilabel_list),
            'multiclass': self.__build_label_codemap(multiclass_list)
        }

        return sets, codes

    @property
    def frame_labels_set(self) -> list[str]:
        """Frame-level label names."""
        return self.frame_lsets['multilabel']

    @property
    def image_labels_set(self) -> list[str]:
        """Image-level label names."""
        return self.image_lsets.get('multilabel', [])

    @property
    def image_categories_set(self) -> list[tuple[str, str]]:
        """Image-level classification category names/values."""
        return self.image_lsets.get('multiclass', [])

    @property
    def segmentation_labels_set(self) -> list[str]:
        """Segmentation label names."""
        return self.seglabel_list

    def _infer_segmentation_group(self) -> tuple[list[str], dict[str, int]]:
        """Infer segmentation labels from annotations when no project is provided."""
        seglabel_set: set[str] = set()

        for annotations in self.resource_annotations:
            for ann in annotations:
                if ann.annotation_type == 'segmentation':
                    # Extract label from the segmentation annotation
                    # Assuming segmentation annotations have an identifier field
                    if hasattr(ann, 'identifier') and ann.identifier:
                        seglabel_set.add(ann.identifier)

        seglabel_list = sorted(seglabel_set)
        seglabel2code = {label: idx + 1 for idx, label in enumerate(seglabel_list)}

        return seglabel_list, seglabel2code

    def _process_segmentation_group(self, groups: dict) -> tuple[list[str], dict[str, int]]:
        """Get segmentation labels from the server."""
        try:
            # groups = self._api.annotationsets.get_segmentation_group(self.project.worklist_id)['groups']

            if not groups:
                return [], {}

            max_index = max([g['index'] for g in groups.values()])
            seglabel_list: list[str] = ['UNKNOWN'] * max_index

            for segname, g in groups.items():
                seglabel_list[g['index'] - 1] = segname

            seglabel2code = {label: idx + 1 for idx, label in enumerate(seglabel_list)}
            return seglabel_list, seglabel2code
        except Exception as e:
            _LOGGER.warning(f"Failed to fetch segmentation labels: {e}")
            return [], {}

    def _preprocess_image_array(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image array to have a consistent dtype.

        Args:
            img: Input image array.

        Returns:
            Preprocessed image array as uint8 or original dtype.
        """
        if img.dtype == np.uint16:
            if not self._logged_uint16_conversion:
                _LOGGER.warning("Converting uint16 to float with normalization to [0, 1]."
                                " If this is not desired, please process the images accordingly by"
                                " either converting to uint8 or float32 beforehand or overriding `_preprocess_image_array`.")
                self._logged_uint16_conversion = True
            img = img.astype(np.float32)
            min_val = img.min()
            img = (img - min_val) / (img.max() - min_val) * 255
            img = img.astype(np.uint8)

        # if not img.flags.writeable:
        #     img = img.copy()
        # img_tensor = torch.from_numpy(img).contiguous()

        # if isinstance(img_tensor, torch.ByteTensor):
        #     img_tensor = img_tensor.to(dtype=torch.get_default_dtype()).div(255)

        return img

    def _process_segmentations(self,
                               segmentations: dict,
                               seg_labels: dict,
                               output_shape: tuple | None = None) -> tuple[Tensor | np.ndarray | dict, dict | None]:
        """
        Process segmentations by optionally converting to semantic format and applying merge strategy.

        Args:
            segmentations: Dict of annotator_id -> segmentation array (num_instances, depth, H, W).
            seg_labels: Dict of annotator_id -> list of label names corresponding to each instance in the segmentation array.
            output_shape: Fallback output shape to use when outputting as semantic segmentation and no segmentations are present to infer from. Should NOT have the number of classes dimension. Example: (depth, H, W)
        """
        # segmentations['author'] shape: (#instances, depth, H, W)
        if self.return_as_semantic_segmentation:
            sem_segs = {}
            for author in segmentations:
                sem_segs[author] = self.annotation_processor.instance_to_semantic_segmentation(
                    segmentations[author], seg_labels[author],
                    num_labels=len(self.segmentation_labels_set)
                )
            segmentations = sem_segs
            _LOGGER.debug(
                f'Converted to semantic segmentation. Shapes: {[segmentations[a].shape for a in segmentations]}')
            if self.semantic_seg_merge_strategy:
                if len(segmentations) > 0:
                    segmentations = self.annotation_processor.apply_merge_strategy(
                        segmentations,
                        strategy=self.semantic_seg_merge_strategy
                    )
                    _LOGGER.debug(f"Merged segmentation shape: {segmentations.shape}")
                else:
                    if output_shape is None:
                        raise ValueError("output_shape must be provided when no segmentations are present"
                                         " to infer shape from.")
                    # Create empty semantic segmentation with just background class
                    segmentations = torch.zeros((len(self.segmentation_labels_set)+1, *output_shape),
                                                dtype=torch.get_default_dtype())
                    segmentations[0] = 1  # background
                    _LOGGER.debug("No segmentations found. "
                                  f"Created empty semantic segmentation with shape: {segmentations.shape}")

                # In semantic format, we don't need `seg_labels`, as the label info is at the new dimension (axis=0) of the semantic segmentation array.
                seg_labels = None

        return segmentations, seg_labels

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get item with full processing."""
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds")

        result = self._get_raw_item(index)

        img = result['image']
        if isinstance(img, np.ndarray):
            img = self._preprocess_image_array(img)
        annotations = result['annotations']
        # resource = result['resource']
        # _LOGGER.debug(f"Loaded image {resource.filename} with shape {img.shape}")

        # Process segmentations
        if self.return_segmentations:
            seg_anns = AnnotationProcessor.filter_annotations(annotations,
                                                              type='segmentation',
                                                              scope='all')
            segmentations, seg_labels, _ = self.annotation_processor.load_segmentations(seg_anns)
            # Apply albumentations if present
            if self.alb_transform:
                aug_result = self.apply_alb_transform(img, segmentations)
                img = aug_result['image']
                result['image'] = img
                segmentations = aug_result['segmentations']

            segmentations, seg_labels = self._process_segmentations(segmentations, seg_labels,
                                                                    output_shape=img.shape[1:])

            result['segmentations'] = segmentations
            if seg_labels:
                result['seg_labels'] = seg_labels

        # Process image-level labels
        result['image_labels'] = self._extract_image_labels(annotations)
        result['image_categories'] = self._extract_image_categories(annotations)

        return result

    @abstractmethod
    def apply_alb_transform(
        self,
        img: np.ndarray,
        segmentations: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        """Apply albumentations transform to image and masks.

        Returns:
            Dict with transformed 'image' and 'segmentations' (dict).
                It is recommended that 'image' has shape (C, depth, H, W)
                and each segmentation of 'segmentations' has shape (num_instances, depth, H, W), so that
                common downstream processing can be applied.
                If not, please override :py:meth:`_process_segmentations` accordingly.

        """
        pass

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.resources)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over dataset."""
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other: 'DatamintBaseDataset') -> ConcatDataset:
        """Concatenate datasets."""
        return ConcatDataset([self, other])  # type: ignore[list-item]

    def build_mlflow_dataset(self) -> 'DatamintMLflowDataset':
        """Create a :class:`~datamint.mlflow.data.DatamintMLflowDataset` for this dataset.

        Args:
            split: The split name (e.g. ``'train'``, ``'val'``, ``'test'``).
        """
        from datamint.mlflow.data import DatamintMLflowDataset

        project = getattr(self, 'project', None)
        project_name = getattr(project, 'name', 'unknown') if project is not None else 'unknown'
        project_id = getattr(project, 'id', 'unknown') if project is not None else 'unknown'

        extra_params = {
            'return_as_semantic_segmentation': self.return_as_semantic_segmentation,
            'semantic_seg_merge_strategy': str(self.semantic_seg_merge_strategy),
            'include_unannotated': self.include_unannotated,
            'include_annotators': self.include_annotators,
            'exclude_annotators': self.exclude_annotators,
            'include_segmentation_names': self.include_segmentation_names,
            'exclude_segmentation_names': self.exclude_segmentation_names,
            'include_image_label_names': self.include_image_label_names,
            'exclude_image_label_names': self.exclude_image_label_names,
            'include_frame_label_names': self.include_frame_label_names,
            'exclude_frame_label_names': self.exclude_frame_label_names,
            'split_source': self.split_source,
            'split_as_of_timestamp': self.split_as_of_timestamp,
        }
        return DatamintMLflowDataset(
            project_id=project_id,
            project_name=project_name,
            split=self.split_name,
            resources=self.resources,
            extra_params=extra_params
        )

    def get_dataloader(self, *args, **kwargs) -> DataLoader:
        """Get DataLoader with proper collate function."""
        return DataLoader(self, *args, collate_fn=self.get_collate_fn(), **kwargs)  # type: ignore[arg-type]

    def get_collate_fn(self) -> Callable[[list[dict]], dict]:
        """Get collate function for DataLoader."""
        def collate_fn(batch: list[dict]) -> dict:
            if not batch:
                return {}

            keys = batch[0].keys()
            collated = {}

            for key in keys:
                values = [item[key] for item in batch]

                if isinstance(values[0], torch.Tensor):
                    shapes = [t.shape for t in values]
                    if all(s == shapes[0] for s in shapes):
                        collated[key] = torch.stack(values)
                    else:
                        _LOGGER.warning(f"Different shapes for {key}: {shapes}")
                        collated[key] = values
                elif isinstance(values[0], np.ndarray):
                    shapes = [a.shape for a in values]
                    if all(s == shapes[0] for s in shapes):
                        collated[key] = np.stack(values)
                    else:
                        _LOGGER.warning(f"Different shapes for {key}: {shapes}")
                        collated[key] = values
                else:
                    collated[key] = values

            return collated

        return collate_fn

    def subset(self, indices: list[int]) -> 'DatamintBaseDataset':
        """Create a dataset subset by slicing resources and annotations."""
        import copy
        new_ds = copy.copy(self)
        try:
            new_ds.resources = [self.resources[i] for i in indices]
            new_ds.resource_annotations = [self.resource_annotations[i] for i in indices]
        except IndexError as e:
            raise IndexError(f"Subset indices out of bounds for dataset of length {len(self)}.") from e
        return new_ds

    def __repr__(self) -> str:
        name = self.project.name if self.project else "<Custom>"
        head = f"Dataset {name}"
        body = [f"Number of datapoints: {len(self)}"]
        if self.split_name is not None:
            body.append(f"Split: {self.split_name}")
        if self.split_source is not None:
            body.append(f"Split source: {self.split_source}")
        if self.split_as_of_timestamp is not None:
            body.append(f"Split as of: {self.split_as_of_timestamp}")

        # if self.manager.root is not None:
        #    body.append(f"Location: {self.manager.dataset_dir}")

        filters = [
            (self.include_annotators, "Including annotators"),
            (self.exclude_annotators, "Excluding annotators"),
            (self.include_segmentation_names, "Including segmentations"),
            (self.exclude_segmentation_names, "Excluding segmentations"),
            (self.include_image_label_names, "Including image labels"),
            (self.exclude_image_label_names, "Excluding image labels"),
            (self.include_frame_label_names, "Including frame labels"),
            (self.exclude_frame_label_names, "Excluding frame labels"),
        ]

        for value, desc in filters:
            if value is not None:
                body.append(f"{desc}: {value}")

        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)

    def split(
        self,
        *,
        seed: int | None = None,
        use_server_splits: bool | None = None,
        use_project_splits: bool | None = None,
        as_of_timestamp: str | None = None,
        **splits: float,
    ) -> dict[str, 'DatamintBaseDataset']:
        """Split the dataset into multiple named subsets.

        The mode is selected automatically when no explicit split mode is
        given:

        - If ratio kwargs are provided (e.g. ``train=0.7``), local splitting
          is used.
        - If no ratio kwargs are provided and the dataset was loaded from a
          project, project-scoped split assignments are used.
        - Otherwise, server-side ``split:*`` tags on resources are used.

        Examples::

            # Local split
            parts = dataset.split(train=0.7, val=0.15, test=0.15, seed=42)
            train_ds = parts['train']

            # Project-scoped split — inferred for project-backed datasets
            parts = dataset.split()

            # Explicit override
            parts = dataset.split(use_project_splits=True)

        Args:
            seed: Random seed for reproducible local splitting.
            use_project_splits: If ``True``, read split assignments from the
                project splits API. If ``None`` (default), project-backed
                datasets prefer this mode when no ratios are provided.
            as_of_timestamp: Historical timestamp to resolve project-scoped
                splits against. When omitted for project-scoped splits, the
                current UTC timestamp is captured and stored on the resolved
                split datasets for later reuse.
            use_server_splits: (DEPRECATED in favor of ``use_project_splits``)
            **splits: Named split ratios (e.g. ``train=0.7, test=0.3``).
                Must sum to 1.0 (±0.01 tolerance). Must be empty when
                *use_server_splits* or *use_project_splits* is ``True``.

        Returns:
            Dictionary mapping split names to new dataset instances.

        Raises:
            ValueError: If ratios are invalid or arguments conflict.
        """
        if use_project_splits is None and use_server_splits is None and not splits:
            use_project_splits = getattr(self, 'project', None) is not None or as_of_timestamp is not None

        if use_project_splits:
            return self._split_by_project_api(splits, as_of_timestamp=as_of_timestamp)

        if as_of_timestamp is not None:
            raise ValueError(
                'as_of_timestamp is only supported with project-scoped splits. '
                'Set use_project_splits=True or use a project-backed dataset with no ratio kwargs.'
            )

        if use_server_splits is None:
            use_server_splits = not splits  # True when no ratios given

        if use_server_splits:
            import warnings
            warnings.warn("use_server_splits and splitting by resource tags are deprecated in favor of use_project_splits. "
                          "Please migrate to project-scoped splits for better reproducibility and management.", DeprecationWarning)
            return self._split_by_server_tags(splits)

        return self._split_locally(splits, seed)

    def _split_by_project_api(
        self,
        splits: dict[str, float],
        as_of_timestamp: str | None = None,
    ) -> dict[str, 'DatamintBaseDataset']:
        """Group resources by project-scoped split assignments from the API."""
        if splits:
            raise ValueError(
                'Ratio kwargs must not be provided with use_project_splits=True.'
            )

        project = getattr(self, 'project', None)
        if project is None:
            raise ValueError(
                'Dataset must be loaded from a project to use use_project_splits=True.'
            )

        resolved_as_of_timestamp = as_of_timestamp or self._utc_now_isoformat()
        assignments = self._api.projects.get_splits(
            project,
            as_of_timestamp=as_of_timestamp,
        )
        resource_split_map = {assignment.resource_id: assignment.split_name for assignment in assignments}
        from collections import defaultdict
        split_indices: dict[str, list[int]] = defaultdict(list)

        for idx, resource in enumerate(self.resources):
            split_name = resource_split_map.get(resource.id)
            if split_name is not None:
                split_indices[split_name].append(idx)

        if not split_indices:
            raise ValueError(
                'No split assignments found for this project. '
                'Call api.projects.assign_splits() first.'
            )

        result = {name: self.subset(indices) for name, indices in split_indices.items()}
        for name, ds in result.items():
            ds.split_name = name
            ds.split_source = 'project_api'
            ds.split_as_of_timestamp = resolved_as_of_timestamp
        return result

    def _split_by_server_tags(
        self,
        splits: dict[str, float],
    ) -> dict[str, 'DatamintBaseDataset']:
        """Group resources by ``split:<name>`` tags."""
        if splits:
            raise ValueError(
                "Ratio kwargs (e.g. train=0.7) must not be provided when "
                "use_server_splits=True."
            )

        from collections import defaultdict
        split_indices: dict[str, list[int]] = defaultdict(list)

        for idx, resource in enumerate(self.resources):
            tags = resource.tags or []
            for tag in tags:
                if tag.startswith("split:"):
                    split_name = tag[len("split:"):]
                    split_indices[split_name].append(idx)

        if not split_indices:
            raise ValueError(
                "No resources have 'split:*' tags. Tag resources on the "
                "server first or use local splitting (use_server_splits=False)."
            )

        result = {name: self.subset(indices) for name, indices in split_indices.items()}
        for name, ds in result.items():
            ds.split_name = name
            ds.split_source = 'server_tags'
            ds.split_as_of_timestamp = None
        return result

    def _split_locally(
        self,
        splits: dict[str, float],
        seed: int | None,
    ) -> dict[str, 'DatamintBaseDataset']:
        """Randomly partition resources by ratios."""
        if len(splits) < 2:
            raise ValueError("At least 2 splits are required (e.g. train=0.7, test=0.3).")

        for name, ratio in splits.items():
            if ratio <= 0:
                raise ValueError(f"Split ratio for '{name}' must be positive, got {ratio}.")

        total = sum(splits.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total:.4f}). "
                f"Provided: {splits}"
            )

        import random
        n = len(self)
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)

        result: dict[str, DatamintBaseDataset] = {}
        start = 0
        split_items = list(splits.items())

        for i, (name, ratio) in enumerate(split_items):
            if i == len(split_items) - 1:
                # Last split gets all remaining indices to avoid rounding issues
                end = n
            else:
                end = start + round(ratio * n)
            result[name] = self.subset(indices[start:end])
            result[name].split_name = name
            result[name].split_source = 'local'
            result[name].split_as_of_timestamp = None
            start = end

        return result

    def filter(
        self,
        *,
        tags: list[str] | None = None,
        filename_pattern: str | None = None,
        has_annotations: bool | None = None,
        annotation_names: list[str] | None = None,
        custom_fn: 'Callable[[Resource, Sequence[Annotation]], bool] | None' = None,
    ) -> 'DatamintBaseDataset':
        """Return a new dataset containing only resources that match **all**
        specified criteria.

        This method is chainable — the returned dataset supports the same
        interface, so you can write::

            filtered = dataset.filter(tags=['busi']).filter(has_annotations=True)

        or combine with :meth:`split`::

            parts = dataset.filter(tags=['ultrasound']).split(train=0.8, test=0.2)

        Args:
            tags: Keep resources whose tags contain **any** of the given values.
            filename_pattern: Keep resources whose filename matches this
                pattern (interpreted as :func:`fnmatch.fnmatch` glob).
            has_annotations: If ``True``, keep only resources with at least one
                annotation.  If ``False``, keep only those **without**
                annotations.
            annotation_names: Keep resources that have at least one annotation
                whose ``identifier`` is in this list.
            custom_fn: Arbitrary predicate receiving ``(resource, annotations)``
                and returning ``True`` to keep the resource.

        Returns:
            A new :class:`DatamintBaseDataset` containing only the matching
            resources.

        Raises:
            ValueError: If no filter criteria are specified.
        """
        if all(v is None for v in (tags, filename_pattern, has_annotations,
                                   annotation_names, custom_fn)):
            raise ValueError("At least one filter criterion must be specified.")

        import fnmatch

        passing_indices: list[int] = []

        for idx, (resource, annotations) in enumerate(
            zip(self.resources, self.resource_annotations)
        ):
            # --- tags (OR within this criterion) ---
            if tags is not None:
                resource_tags = resource.tags or []
                if not any(t in resource_tags for t in tags):
                    continue

            # --- filename_pattern ---
            if filename_pattern is not None:
                if not fnmatch.fnmatch(resource.filename, filename_pattern):
                    continue

            # --- has_annotations ---
            if has_annotations is not None:
                has_any = len(annotations) > 0
                if has_any != has_annotations:
                    continue

            # --- annotation_names ---
            if annotation_names is not None:
                ann_identifiers = {a.identifier for a in annotations}
                if not ann_identifiers.intersection(annotation_names):
                    continue

            # --- custom_fn ---
            if custom_fn is not None:
                if not custom_fn(resource, annotations):
                    continue

            passing_indices.append(idx)

        return self.subset(passing_indices)
