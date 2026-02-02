"""
DatamintBaseDataset - Abstract base class for all Datamint datasets.

Provides the PyTorch Dataset interface with transform support and annotation
filtering, while delegating data management to DatamintProjectManager.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence, Callable, Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from datamint.exceptions import DatamintException
from datamint.entities import Annotation
from datamint import Api
from .annotation_processor import AnnotationProcessor, MergeStrategy

if TYPE_CHECKING:
    from datamint.entities import Resource, Project

_LOGGER = logging.getLogger(__name__)


class DatamintDatasetException(DatamintException):
    """Exception raised for dataset errors."""
    pass


class DatamintBaseDataset(ABC):
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
    """

    resources: Sequence['Resource']
    resource_annotations: Sequence[Sequence[Annotation]]
    project: 'Project | None'

    def __init__(
        self,
        project: 'str | Project | None' = None,
        resources: 'Sequence[Resource] | Sequence[str] | None' = None,
        auto_update: bool = True,
        api_key: str | None = None,
        server_url: str | None = None,
        # all_annotations: bool = False,
        return_metainfo: bool = True,
        return_segmentations: bool = True,
        return_as_semantic_segmentation: bool = False,
        semantic_seg_merge_strategy: MergeStrategy | None = None,
        alb_transform: Callable | None = None,
        include_unannotated: bool = True,
        include_annotators: list[str] | None = None,
        exclude_annotators: list[str] | None = None,
        include_segmentation_names: list[str] | None = None,
        exclude_segmentation_names: list[str] | None = None,
        include_image_label_names: list[str] | None = None,
        exclude_image_label_names: list[str] | None = None,
        include_frame_label_names: list[str] | None = None,
        exclude_frame_label_names: list[str] | None = None,
    ):
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

        # Initialize API
        self._api = Api(
            server_url=server_url,
            api_key=api_key,
            check_connection=auto_update
        )

        # Initialize from project or resources
        if resources is not None:
            self.resources = self._initialize_from_resources(resources, self._api)
            self.project = None
        else:
            self.project, self.resources = self._initialize_from_project(project, self._api)  # type: ignore

        # Fetch annotations
        self.resource_annotations = list(self._api.annotations.get_list(
            resource=self.resources,
            group_by_resource=True,
        ))

        # Store configuration
        self.return_metainfo = return_metainfo
        self.return_segmentations = return_segmentations
        self.return_as_semantic_segmentation = return_as_semantic_segmentation
        self.semantic_seg_merge_strategy: MergeStrategy | None = semantic_seg_merge_strategy
        self.include_unannotated = include_unannotated

        # Transforms
        self.alb_transform = alb_transform

        # Filtering
        self.include_annotators = include_annotators
        self.exclude_annotators = exclude_annotators
        self.include_segmentation_names = include_segmentation_names
        self.exclude_segmentation_names = exclude_segmentation_names
        self.include_image_label_names = include_image_label_names
        self.exclude_image_label_names = exclude_image_label_names
        self.include_frame_label_names = include_frame_label_names
        self.exclude_frame_label_names = exclude_frame_label_names

        # Internal state
        self._logged_uint16_conversion = False

        # Setup
        self._setup_dataset()

    def _extract_image_labels(
        self,
        annotations: Sequence[Annotation],
    ) -> dict[str, torch.Tensor]:
        """Extract image-level label annotations.

        Args:
            annotations: All annotations for the item.

        Returns:
            Dict of annotator_id -> label tensor.
        """
        label_annotations = AnnotationProcessor.filter_annotations(
            annotations, type='label', scope='image'
        )
        return self.annotation_processor.convert_image_labels(label_annotations)

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
        api: Api
    ) -> tuple['Project', list['Resource']]:
        """Initialize dataset from a project (name or object)."""

        # Handle Project object vs string
        if isinstance(project, str):
            project = api.projects.get_by_name(project)
            if project is None:
                raise DatamintDatasetException(f"Project '{project}' not found.")
        else:
            # Attach API to project if not already set
            if not hasattr(project, '_api') or project._api is None:
                project._api = api.projects

        # Fetch resources
        resources = list(project.fetch_resources())
        return project, resources

    def _initialize_from_resources(
        self,
        resources: 'Sequence[Resource] | Sequence[str]',
        api: Api
    ) -> list['Resource']:
        """Initialize dataset from a list of resources."""
        # Normalize resources (handle IDs vs objects)
        resource_list: list[Resource] = []
        if resources:
            first_item = resources[0] if len(resources) > 0 else None
            if isinstance(first_item, str):
                # Fetch Resource objects from IDs
                resource_list = [api.resources.get_by_id(rid) for rid in resources]  # type: ignore
            else:
                resource_list = list(resources)  # type: ignore

        # Attach API to resources if needed
        for res in resource_list:
            if not hasattr(res, '_api') or res._api is None:
                res._api = api.resources

        return resource_list

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
        if not self.include_unannotated:
            self._filter_unannotated()

    def _setup_labels(self) -> None:
        """Setup label sets and mappings."""
        # Frame and image labels
        self.frame_lsets, self.frame_lcodes = self._get_labels_set(framed=True)
        self.image_lsets, self.image_lcodes = self._get_labels_set(framed=False)

        # Segmentation labels
        self.seglabel_list, self.seglabel2code = self._get_segmentation_labels()

    def _setup_annotation_processor(self) -> None:
        """Initialize the annotation processor.

        Calls _create_annotation_processor() which subclasses can override
        to return the appropriate processor type.
        """
        self.annotation_processor = AnnotationProcessor(
            seglabel2code=self.seglabel2code,
            image_labels_set=self.image_labels_set,
            image_lcodes=self.image_lcodes,
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

    def _filter_annotations(self, annotations: Sequence[Annotation]) -> list[Annotation]:
        """Filter annotations based on include/exclude settings."""
        return [ann for ann in annotations if self._should_include_annotation(ann)]

    def _should_include_annotation(self, ann: Annotation) -> bool:
        """Check if annotation should be included."""
        # Check annotator
        annotator = ann.created_by
        if annotator is not None and not self._should_include_annotator(annotator):
            return False

        # Check by annotation type
        if ann.annotation_type == 'segmentation':
            return self._should_include_segmentation(ann.identifier)
        elif ann.annotation_type == 'label':
            if ann.frame_index is None:  # image-level
                return self._should_include_image_label(ann.identifier)
            else:  # frame-level
                return self._should_include_frame_label(ann.identifier)

        return True

    def _should_include_annotator(self, annotator_id: str) -> bool:
        if self.include_annotators is not None:
            return annotator_id in self.include_annotators
        if self.exclude_annotators is not None:
            return annotator_id not in self.exclude_annotators
        return True

    def _should_include_segmentation(self, name: str) -> bool:
        if self.include_segmentation_names is not None:
            return name in self.include_segmentation_names
        if self.exclude_segmentation_names is not None:
            return name not in self.exclude_segmentation_names
        return True

    def _should_include_image_label(self, name: str) -> bool:
        if self.include_image_label_names is not None:
            return name in self.include_image_label_names
        if self.exclude_image_label_names is not None:
            return name not in self.exclude_image_label_names
        return True

    def _should_include_frame_label(self, name: str) -> bool:
        if self.include_frame_label_names is not None:
            return name in self.include_frame_label_names
        if self.exclude_frame_label_names is not None:
            return name not in self.exclude_frame_label_names
        return True

    def _get_labels_set(self, framed: bool) -> tuple[dict, dict[str, dict[str, int]]]:
        """Get label sets and codes."""
        scope = 'frame' if framed else 'image'
        multilabel_set: set[str] = set()
        multiclass_set: set[tuple[str, Any]] = set()

        for annotations in self.resource_annotations:
            for ann in annotations:
                ann_scope = 'image' if ann.frame_index is None else 'frame'
                if ann_scope != scope:
                    continue

                if ann.annotation_type == 'label':
                    multilabel_set.add(ann.identifier)
                elif ann.annotation_type == 'category':
                    multiclass_set.add((ann.identifier, ann.value))

        multilabel_list = sorted(multilabel_set)
        multiclass_list = sorted(multiclass_set)

        sets = {
            'multilabel': multilabel_list,
            'multiclass': multiclass_list
        }
        codes = {
            'multilabel': {label: idx for idx, label in enumerate(multilabel_list)},
            'multiclass': {label: idx for idx, label in enumerate(multiclass_list)}
        }

        return sets, codes

    @property
    def frame_labels_set(self) -> list[str]:
        """Frame-level label names."""
        return self.frame_lsets['multilabel']

    @property
    def image_labels_set(self) -> list[str]:
        """Image-level label names."""
        return self.image_lsets['multilabel']

    @property
    def segmentation_labels_set(self) -> list[str]:
        """Segmentation label names."""
        return self.seglabel_list

    def _get_segmentation_labels(self) -> tuple[list[str], dict[str, int]]:
        """Get segmentation labels from the server."""
        try:
            worklist_id = getattr(self.project, 'worklist_id', None)
            groups: dict[str, dict] = self._api.annotationsets.get_segmentation_group(worklist_id)['groups']

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
                               seg_labels: dict) -> tuple[Tensor | np.ndarray | dict, dict | None]:
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
                if segmentations:
                    segmentations = self.annotation_processor.apply_merge_strategy(
                        segmentations,
                        strategy=self.semantic_seg_merge_strategy
                    )
                    _LOGGER.debug(f"Merged segmentation shape: {segmentations.shape}")

                seg_labels = None
                _LOGGER.debug(f'merged segmentations. Final shape: {segmentations.shape}')

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
        resource = result['resource']
        _LOGGER.debug(f"Loaded image {resource.filename} with shape {img.shape}")
        _LOGGER.debug(f"Annotations: {len(annotations)} found")

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
                _LOGGER.debug(
                    f"Applied albumentations transform. Image shape: {img.shape} and segs shape: {[segmentations[a].shape for a in segmentations]}")

            segmentations, seg_labels = self._process_segmentations(segmentations, seg_labels)

            result['segmentations'] = segmentations
            if seg_labels:
                result['seg_labels'] = seg_labels

        # Process image-level labels
        result['image_labels'] = self._extract_image_labels(annotations)

        return result

    @abstractmethod
    def apply_alb_transform(
        self,
        img: np.ndarray,
        segmentations: dict[str, np.ndarray]
    ) -> dict[str, Any]:
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

    def subset(self, indices: list[int]) -> 'DatamintBaseDataset':
        pass

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
                    collated[key] = np.stack(values)
                else:
                    collated[key] = values

            return collated

        return collate_fn

    def __repr__(self) -> str:
        name = self.project.name if self.project else "<Custom>"
        head = f"Dataset {name}"
        body = [f"Number of datapoints: {len(self)}"]

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
