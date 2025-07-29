import os
import requests
from tqdm import tqdm
from typing import Optional, Callable, Any, Literal
import logging
import shutil
import json
import yaml
import pydicom
from pydicom.dataset import FileDataset
import numpy as np
from datamint import configs
from torch.utils.data import DataLoader
import torch
from torch import Tensor
from datamint.apihandler.base_api_handler import DatamintException
from medimgkit.dicom_utils import is_dicom
import cv2
from medimgkit.io_utils import read_array_normalized
from datetime import datetime

_LOGGER = logging.getLogger(__name__)


class DatamintDatasetException(DatamintException):
    pass


class DatamintBaseDataset:
    """Class to download and load datasets from the Datamint API.

    Args:
        project_name: Name of the project to download.
        root: Root directory of dataset where data already exists or will be downloaded.
        auto_update: If True, the dataset will be checked for updates and downloaded if necessary.
        api_key: API key to access the Datamint API. If not provided, it will look for the
            environment variable 'DATAMINT_API_KEY'. Not necessary if
            you don't want to download/update the dataset.
        return_dicom: If True, the DICOM object will be returned, if the image is a DICOM file.
        return_metainfo: If True, the metainfo of the image will be returned.
        return_annotations: If True, the annotations of the image will be returned.
        return_frame_by_frame: If True, each frame of a video/DICOM/3d-image will be returned separately.
        include_unannotated: If True, images without annotations will be included.
        all_annotations: If True, all annotations will be downloaded, including the ones that are not set as closed/done.
        server_url: URL of the Datamint server. If not provided, it will use the default server.
        include_annotators: List of annotators to include. If None, all annotators will be included.
        exclude_annotators: List of annotators to exclude. If None, no annotators will be excluded.
        include_segmentation_names: List of segmentation names to include. If None, all segmentations will be included.
        exclude_segmentation_names: List of segmentation names to exclude. If None, no segmentations will be excluded.
        include_image_label_names: List of image label names to include. If None, all image labels will be included.
        exclude_image_label_names: List of image label names to exclude. If None, no image labels will be excluded.
        include_frame_label_names: List of frame label names to include. If None, all frame labels will be included.
        exclude_frame_label_names: List of frame label names to exclude. If None, no frame labels will be excluded.
    """

    DATAMINT_DEFAULT_DIR = ".datamint"
    DATAMINT_DATASETS_DIR = "datasets"

    def __init__(
        self,
        project_name: str,
        root: str | None = None,
        auto_update: bool = True,
        api_key: Optional[str] = None,
        server_url: Optional[str] = None,
        return_dicom: bool = False,
        return_metainfo: bool = True,
        return_annotations: bool = True,
        return_frame_by_frame: bool = False,
        include_unannotated: bool = True,
        all_annotations: bool = False,
        # Filtering parameters
        include_annotators: Optional[list[str]] = None,
        exclude_annotators: Optional[list[str]] = None,
        include_segmentation_names: Optional[list[str]] = None,
        exclude_segmentation_names: Optional[list[str]] = None,
        include_image_label_names: Optional[list[str]] = None,
        exclude_image_label_names: Optional[list[str]] = None,
        include_frame_label_names: Optional[list[str]] = None,
        exclude_frame_label_names: Optional[list[str]] = None,
    ):
        self._validate_inputs(project_name, include_annotators, exclude_annotators,
                             include_segmentation_names, exclude_segmentation_names,
                             include_image_label_names, exclude_image_label_names,
                             include_frame_label_names, exclude_frame_label_names)
        
        self._initialize_config(
            project_name, auto_update, all_annotations, return_dicom,
            return_metainfo, return_annotations, return_frame_by_frame,
            include_unannotated, include_annotators, exclude_annotators,
            include_segmentation_names, exclude_segmentation_names,
            include_image_label_names, exclude_image_label_names,
            include_frame_label_names, exclude_frame_label_names
        )
        
        self._setup_api_handler(server_url, api_key, auto_update)
        self._setup_directories(root)
        self._setup_dataset()
        self._post_process_data()

    def _validate_inputs(
        self,
        project_name: str,
        include_annotators: Optional[list[str]],
        exclude_annotators: Optional[list[str]],
        include_segmentation_names: Optional[list[str]],
        exclude_segmentation_names: Optional[list[str]],
        include_image_label_names: Optional[list[str]],
        exclude_image_label_names: Optional[list[str]],
        include_frame_label_names: Optional[list[str]],
        exclude_frame_label_names: Optional[list[str]],
    ) -> None:
        """Validate input parameters."""
        if project_name is None:
            raise ValueError("project_name is required.")

        # Validate mutually exclusive filtering parameters
        filter_pairs = [
            (include_annotators, exclude_annotators, "annotators"),
            (include_segmentation_names, exclude_segmentation_names, "segmentation_names"),
            (include_image_label_names, exclude_image_label_names, "image_label_names"),
            (include_frame_label_names, exclude_frame_label_names, "frame_label_names"),
        ]
        
        for include_param, exclude_param, param_name in filter_pairs:
            if include_param is not None and exclude_param is not None:
                raise ValueError(f"Cannot set both include_{param_name} and exclude_{param_name} at the same time")

    def _initialize_config(
        self,
        project_name: str,
        auto_update: bool,
        all_annotations: bool,
        return_dicom: bool,
        return_metainfo: bool,
        return_annotations: bool,
        return_frame_by_frame: bool,
        include_unannotated: bool,
        include_annotators: Optional[list[str]],
        exclude_annotators: Optional[list[str]],
        include_segmentation_names: Optional[list[str]],
        exclude_segmentation_names: Optional[list[str]],
        include_image_label_names: Optional[list[str]],
        exclude_image_label_names: Optional[list[str]],
        include_frame_label_names: Optional[list[str]],
        exclude_frame_label_names: Optional[list[str]],
    ) -> None:
        """Initialize configuration parameters."""
        self.project_name = project_name
        self.all_annotations = all_annotations
        self.return_dicom = return_dicom
        self.return_metainfo = return_metainfo
        self.return_frame_by_frame = return_frame_by_frame
        self.return_annotations = return_annotations
        self.include_unannotated = include_unannotated
        self.discard_without_annotations = not include_unannotated

        # Filtering parameters
        self.include_annotators = include_annotators
        self.exclude_annotators = exclude_annotators
        self.include_segmentation_names = include_segmentation_names
        self.exclude_segmentation_names = exclude_segmentation_names
        self.include_image_label_names = include_image_label_names
        self.exclude_image_label_names = exclude_image_label_names
        self.include_frame_label_names = include_frame_label_names
        self.exclude_frame_label_names = exclude_frame_label_names

        # Internal state
        self.__logged_uint16_conversion = False

    def _setup_api_handler(self, server_url: Optional[str], api_key: Optional[str], auto_update: bool) -> None:
        """Setup API handler and validate connection."""
        from datamint.apihandler.api_handler import APIHandler

        self.api_handler = APIHandler(
            root_url=server_url, 
            api_key=api_key,
            check_connection=auto_update
        )
        self.server_url = self.api_handler.root_url
        self.api_key = self.api_handler.api_key

        if self.api_key is None:
            _LOGGER.warning(
                "API key not provided. If you want to download data, please provide an API key, "
                f"either by passing it as an argument, "
                f"setting environment variable {configs.ENV_VARS[configs.APIKEY_KEY]} or "
                "using datamint-config command line tool."
            )

    def _setup_directories(self, root: str | None) -> None:
        """Setup root and dataset directories."""
        if root is None:
            root = os.path.join(
                os.path.expanduser("~"),
                self.DATAMINT_DEFAULT_DIR,
                self.DATAMINT_DATASETS_DIR
            )
            os.makedirs(root, exist_ok=True)
        else:
            root = os.path.expanduser(root)
            if not os.path.isdir(root):
                raise NotADirectoryError(f"Root directory not found: {root}")

        self.root = root
        self.dataset_dir = os.path.join(root, self.project_name)
        self.dataset_zippath = os.path.join(root, f'{self.project_name}.zip')

    def _setup_dataset(self) -> None:
        """Setup dataset by downloading or loading existing data."""
        local_dataset_exists = os.path.exists(os.path.join(self.dataset_dir, 'dataset.json'))

        if local_dataset_exists and not hasattr(self, 'project_info'):
            # We might not need project info if not updating
            self.dataset_id = None
        else:
            self.project_info = self.get_info()
            self.dataset_id = self.project_info['dataset_id']

        self._handle_dataset_download_or_update(local_dataset_exists)
        self._load_metadata()

    def _handle_dataset_download_or_update(self, local_dataset_exists: bool) -> None:
        """Handle dataset download or update logic."""
        if local_dataset_exists:
            _LOGGER.info(f"Dataset directory already exists: {self.dataset_dir}")
            # Check for updates if auto_update is enabled and we have API access
            if hasattr(self, 'project_info'):
                _LOGGER.info("Checking for updates...")
                self._check_version()
        else:
            if self.api_key is None:
                raise DatamintDatasetException("API key is required to download the dataset.")
            _LOGGER.info(f"No data found at {self.dataset_dir}. Downloading...")
            self.download_project()

    def _load_metadata(self) -> None:
        """Load and process dataset metadata."""
        if not hasattr(self, 'metainfo'):
            with open(os.path.join(self.dataset_dir, 'dataset.json'), 'r') as file:
                self.metainfo = json.load(file)
        
        self.images_metainfo = self.metainfo['resources']
        self._apply_annotation_filters()

    def _apply_annotation_filters(self) -> None:
        """Apply annotation filters and remove unannotated images if needed."""
        # Filter annotations for each image
        for imginfo in self.images_metainfo:
            imginfo['annotations'] = self._filter_annotations(imginfo['annotations'])

        # Filter out images with no annotations if needed
        if self.discard_without_annotations:
            original_count = len(self.images_metainfo)
            self.images_metainfo = self._filter_items(self.images_metainfo)
            _LOGGER.info(f"Discarded {original_count - len(self.images_metainfo)} images without annotations.")

    def _post_process_data(self) -> None:
        """Post-process data after loading metadata."""
        self._check_integrity()
        self._calculate_dataset_length()
        self._precompute_frame_data()
        self._setup_labels()
        
        if self.discard_without_annotations and self.return_frame_by_frame:
            self._filter_unannotated()

    def _calculate_dataset_length(self) -> None:
        """Calculate the total dataset length based on frame-by-frame setting."""
        if self.return_frame_by_frame:
            self.dataset_length = sum(
                self.read_number_of_frames(os.path.join(self.dataset_dir, imginfo['file']))
                for imginfo in self.images_metainfo
            )
        else:
            self.dataset_length = len(self.images_metainfo)

    def _precompute_frame_data(self) -> None:
        """Precompute frame-related data for efficient indexing."""
        self.num_frames_per_resource = self.__compute_num_frames_per_resource()
        self._cumulative_frames = np.cumsum([0] + self.num_frames_per_resource)
        self.subset_indices = list(range(self.dataset_length))

    def _setup_labels(self) -> None:
        """Setup label sets and mappings."""
        self.frame_lsets, self.frame_lcodes = self._get_labels_set(framed=True)
        self.image_lsets, self.image_lcodes = self._get_labels_set(framed=False)

    def _filter_items(self, images_metainfo: list[dict]) -> list[dict]:
        """Filter items that have annotations."""
        return [img for img in images_metainfo if len(img.get('annotations', []))]

    def _filter_unannotated(self) -> None:
        """Filter out frames that don't have any segmentations."""
        filtered_indices = []
        for i in range(len(self.subset_indices)):
            item_meta = self._get_image_metainfo(i)
            annotations = item_meta.get('annotations', [])

            # Check if there are any segmentation annotations
            has_segmentations = any(ann['type'] == 'segmentation' for ann in annotations)

            if has_segmentations:
                filtered_indices.append(self.subset_indices[i])

        self.subset_indices = filtered_indices
        _LOGGER.info(f"Filtered dataset: {len(self.subset_indices)} frames with segmentations")

    def __compute_num_frames_per_resource(self) -> list[int]:
        """Compute number of frames for each resource."""
        return [
            self.read_number_of_frames(os.path.join(self.dataset_dir, imginfo['file']))
            for imginfo in self.images_metainfo
        ]

    @property
    def frame_labels_set(self) -> list[str]:
        """Returns the set of independent labels in the dataset (multi-label tasks)."""
        return self.frame_lsets['multilabel']

    @property
    def frame_categories_set(self) -> list[tuple[str, str]]:
        """Returns the set of categories in the dataset (multi-class tasks)."""
        return self.frame_lsets['multiclass']

    @property
    def image_labels_set(self) -> list[str]:
        """Returns the set of independent labels in the dataset (multi-label tasks)."""
        return self.image_lsets['multilabel']

    @property
    def image_categories_set(self) -> list[tuple[str, str]]:
        """Returns the set of categories in the dataset (multi-class tasks)."""
        return self.image_lsets['multiclass']

    @property
    def segmentation_labels_set(self) -> list[str]:
        """Returns the set of segmentation labels in the dataset."""
        return self.frame_lsets['segmentation']

    def _get_annotations_internal(
        self,
        annotations: list[dict],
        type: Literal['label', 'category', 'segmentation', 'all'] = 'all',
        scope: Literal['frame', 'image', 'all'] = 'all'
    ) -> list[dict]:
        """Internal method to filter annotations by type and scope."""
        if type not in ['label', 'category', 'segmentation', 'all']:
            raise ValueError(f"Invalid value for 'type': {type}")
        if scope not in ['frame', 'image', 'all']:
            raise ValueError(f"Invalid value for 'scope': {scope}")

        filtered_annotations = []
        for ann in annotations:
            ann_scope = 'image' if ann.get('index', None) is None else 'frame'
            
            type_matches = type == 'all' or ann['type'] == type
            scope_matches = scope == 'all' or scope == ann_scope
            
            if type_matches and scope_matches:
                filtered_annotations.append(ann)
                
        return filtered_annotations

    def get_annotations(
        self,
        index: int,
        type: Literal['label', 'category', 'segmentation', 'all'] = 'all',
        scope: Literal['frame', 'image', 'all'] = 'all'
    ) -> list[dict]:
        """Returns the annotations of the image at the given index.

        Args:
            index: Index of the image.
            type: The type of the annotations. Can be 'label', 'category', 'segmentation' or 'all'.
            scope: The scope of the annotations. Can be 'frame', 'image' or 'all'.

        Returns:
            The annotations of the image.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")
        
        imginfo = self._get_image_metainfo(index)
        return self._get_annotations_internal(imginfo['annotations'], type=type, scope=scope)

    @staticmethod
    def read_number_of_frames(filepath: str) -> int:
        """Read the number of frames in a file."""
        if is_dicom(filepath):
            ds = pydicom.dcmread(filepath)
            return getattr(ds, 'NumberOfFrames', 1)
        elif filepath.lower().endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(filepath)
            try:
                return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
        elif filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            return 1
        else:
            raise ValueError(f"Unsupported file type: {filepath}")

    def get_resources_ids(self) -> list[str]:
        """Get list of resource IDs."""
        return [
            self.__getitem_internal(i, only_load_metainfo=True)['metainfo']['id'] 
            for i in self.subset_indices
        ]

    def _get_labels_set(self, framed: bool) -> tuple[dict, dict[str, dict[str, int]]]:
        """Returns the set of labels and mappings to integers.

        Args:
            framed: If True, get frame-level labels, otherwise image-level labels.

        Returns:
            Tuple containing label sets and label-to-code mappings.
        """
        scope = 'frame' if framed else 'image'

        multilabel_set = set()
        segmentation_labels = set()
        multiclass_set = set()

        for i in range(len(self)):
            # Collect labels by type
            label_anns = self.get_annotations(i, type='label', scope=scope)
            multilabel_set.update(ann['name'] for ann in label_anns)

            seg_anns = self.get_annotations(i, type='segmentation', scope=scope)
            segmentation_labels.update(ann['name'] for ann in seg_anns)

            cat_anns = self.get_annotations(i, type='category', scope=scope)
            multiclass_set.update((ann['name'], ann['value']) for ann in cat_anns)

        # Sort and create mappings
        multilabel_list = sorted(multilabel_set)
        multiclass_list = sorted(multiclass_set)
        segmentation_list = sorted(segmentation_labels)

        sets = {
            'multilabel': multilabel_list,
            'segmentation': segmentation_list,
            'multiclass': multiclass_list
        }
        
        codes_map = {
            'multilabel': {label: idx for idx, label in enumerate(multilabel_list)},
            'segmentation': {label: idx + 1 for idx, label in enumerate(segmentation_list)},
            'multiclass': {label: idx for idx, label in enumerate(multiclass_list)}
        }
        
        return sets, codes_map

    def get_framelabel_distribution(self, normalize: bool = False) -> dict[str, float]:
        """Returns the distribution of frame labels in the dataset."""
        return self._get_label_distribution('label', 'frame', normalize)

    def get_segmentationlabel_distribution(self, normalize: bool = False) -> dict[str, float]:
        """Returns the distribution of segmentation labels in the dataset."""
        return self._get_label_distribution('segmentation', 'all', normalize)

    def _get_label_distribution(self, ann_type: str, scope: str, normalize: bool) -> dict[str, float]:
        """Helper method to calculate label distributions."""
        if ann_type == 'label' and scope == 'frame':
            labels = self.frame_labels_set
        elif ann_type == 'segmentation':
            labels = self.segmentation_labels_set
        else:
            raise ValueError(f"Unsupported combination: type={ann_type}, scope={scope}")

        distribution = {label: 0 for label in labels}
        
        for imginfo in self.images_metainfo:
            for ann in imginfo.get('annotations', []):
                condition_met = (
                    ann['type'] == ann_type and 
                    (scope == 'all' or 
                     (scope == 'frame' and ann.get('index') is not None) or
                     (scope == 'image' and ann.get('index') is None))
                )
                if condition_met and ann['name'] in distribution:
                    distribution[ann['name']] += 1

        if normalize:
            total = sum(distribution.values())
            if total > 0:
                distribution = {k: v / total for k, v in distribution.items()}
                
        return distribution

    def _check_integrity(self) -> None:
        """Check if all image files exist."""
        missing_files = []
        for imginfo in self.images_metainfo:
            filepath = os.path.join(self.dataset_dir, imginfo['file'])
            if not os.path.isfile(filepath):
                missing_files.append(imginfo['file'])
        
        if missing_files:
            raise DatamintDatasetException(f"Image files not found: {missing_files}")

    def _get_datasetinfo(self) -> dict:
        """Get dataset information from API."""
        all_datasets = self.api_handler.get_datasets()

        for dataset in all_datasets:
            if dataset['id'] == self.dataset_id:
                return dataset

        available_datasets = [(d['name'], d['id']) for d in all_datasets]
        raise DatamintDatasetException(
            f"Dataset with id '{self.dataset_id}' not found. "
            f"Available datasets: {available_datasets}"
        )

    def get_info(self) -> dict:
        """Get project information from API."""
        project = self.api_handler.get_project_by_name(self.project_name)
        if 'error' in project:
            available_projects = project['all_projects']
            raise DatamintDatasetException(
                f"Project with name '{self.project_name}' not found. "
                f"Available projects: {available_projects}"
            )
        return project

    def _run_request(self, session, request_args) -> requests.Response:
        response = session.request(**request_args)
        if response.status_code == 400:
            _LOGGER.error(f"Bad request: {response.text}")
        response.raise_for_status()
        return response

    def _get_jwttoken(self, dataset_id, session) -> str:
        if dataset_id is None:
            raise ValueError("Dataset ID is required to download the dataset.")
        request_params = {
            'method': 'GET',
            'url': f'{self.server_url}/datasets/{dataset_id}/download/png',
            'headers': {'apikey': self.api_key},
            'stream': True
        }
        _LOGGER.debug(f"Getting jwt token for dataset {dataset_id}...")
        response = self._run_request(session, request_params)
        progress_bar = None
        number_processed_images = 0

        # check if the response is a stream of data and everything is ok
        if response.status_code != 200:
            msg = f"Getting jwt token failed with status code={response.status_code}: {response.text}"
            raise DatamintDatasetException(msg)

        try:
            response_iterator = response.iter_lines(decode_unicode=True)
            for line in response_iterator:
                line = line.strip()
                if 'event: error' in line:
                    error_msg = line+'\n'
                    error_msg += '\n'.join(response_iterator)
                    raise DatamintDatasetException(f"Getting jwt token failed:\n{error_msg}")
                if not line.startswith('data:'):
                    continue
                dataline = yaml.safe_load(line)['data']
                if 'zip' in dataline:
                    _LOGGER.debug(f"Got jwt token for dataset {dataset_id}")
                    return dataline['zip']  # Function normally ends here
                elif 'processedImages' in dataline:
                    if progress_bar is None:
                        total_size = int(dataline['totalImages'])
                        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
                    processed_images = int(dataline['processedImages'])
                    if number_processed_images < processed_images:
                        progress_bar.update(processed_images - number_processed_images)
                        number_processed_images = processed_images
                else:
                    _LOGGER.warning(f"Unknown data line: {dataline}")
        except Exception as e:
            raise e
        finally:
            if progress_bar is not None:
                progress_bar.close()

        raise DatamintDatasetException("Getting jwt token failed! No dataline with 'zip' entry found.")

    def __repr__(self) -> str:
        """String representation of the dataset."""
        head = f"Dataset {self.project_name}"
        body = [f"Number of datapoints: {self.__len__()}"]
        
        if self.root is not None:
            body.append(f"Location: {self.dataset_dir}")

        # Add filter information
        filter_info = [
            (self.include_annotators, "Including only annotators"),
            (self.exclude_annotators, "Excluding annotators"),
            (self.include_segmentation_names, "Including only segmentations"),
            (self.exclude_segmentation_names, "Excluding segmentations"),
            (self.include_image_label_names, "Including only image labels"),
            (self.exclude_image_label_names, "Excluding image labels"),
            (self.include_frame_label_names, "Including only frame labels"),
            (self.exclude_frame_label_names, "Excluding frame labels"),
        ]
        
        for filter_value, description in filter_info:
            if filter_value is not None:
                body.append(f"{description}: {filter_value}")

        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def download_project(self) -> None:
        """Download project data from API."""
        from torchvision.datasets.utils import extract_archive

        dataset_info = self._get_datasetinfo()
        self.dataset_id = dataset_info['id']
        self.last_updaded_at = dataset_info['updated_at']

        self.api_handler.download_project(
            self.project_info['id'],
            self.dataset_zippath,
            all_annotations=self.all_annotations,
            include_unannotated=self.include_unannotated
        )
        
        _LOGGER.debug("Downloaded dataset")
        
        if os.path.getsize(self.dataset_zippath) == 0:
            raise DatamintDatasetException("Download failed.")

        self._extract_and_update_metadata()

    def _extract_and_update_metadata(self) -> None:
        """Extract downloaded archive and update metadata."""
        from torchvision.datasets.utils import extract_archive
        
        if os.path.exists(self.dataset_dir):
            _LOGGER.info(f"Deleting existing dataset directory: {self.dataset_dir}")
            shutil.rmtree(self.dataset_dir)
            
        extract_archive(self.dataset_zippath, self.dataset_dir, remove_finished=True)
        
        # Load and update metadata
        datasetjson_path = os.path.join(self.dataset_dir, 'dataset.json')
        with open(datasetjson_path, 'r') as file:
            self.metainfo = json.load(file)
            
        self._update_metadata_timestamps()
        
        # Save updated metadata
        with open(datasetjson_path, 'w') as file:
            json.dump(self.metainfo, file)

    def _update_metadata_timestamps(self) -> None:
        """Update metadata with correct timestamps."""
        if 'updated_at' not in self.metainfo:
            self.metainfo['updated_at'] = self.last_updaded_at
        else:
            try:
                local_time = datetime.fromisoformat(self.metainfo['updated_at'])
                server_time = datetime.fromisoformat(self.last_updaded_at)
                
                if local_time < server_time:
                    _LOGGER.warning(
                        f"Inconsistent updated_at dates detected "
                        f"({self.metainfo['updated_at']} < {self.last_updaded_at}). "
                        f"Fixing it to {self.last_updaded_at}"
                    )
                    self.metainfo['updated_at'] = self.last_updaded_at
            except Exception as e:
                _LOGGER.warning(f"Failed to parse updated_at date: {e}")

        self.metainfo['all_annotations'] = self.all_annotations

    def _load_image(self, filepath: str, index: int | None = None) -> tuple[Tensor, FileDataset | None]:
        """Load image from file with optional frame index."""
        if os.path.isdir(filepath):
            raise NotImplementedError("Loading an image from a directory is not supported yet.")

        if self.return_frame_by_frame:
            img, ds = read_array_normalized(filepath, return_metainfo=True, index=index)
        else:
            img, ds = read_array_normalized(filepath, return_metainfo=True)

        img = self._process_image_array(img)
        return img, ds

    def _process_image_array(self, img: np.ndarray) -> Tensor:
        """Process numpy array to tensor with proper normalization."""
        if img.dtype == np.uint16:
            if not self.__logged_uint16_conversion:
                _LOGGER.info("Original image is uint16, converting to uint8")
                self.__logged_uint16_conversion = True

            # Min-max normalization
            img = img.astype(np.float32)
            min_val = img.min()
            img = (img - min_val) / (img.max() - min_val) * 255
            img = img.astype(np.uint8)

        img_tensor = torch.from_numpy(img).contiguous()
        
        if isinstance(img_tensor, torch.ByteTensor):
            img_tensor = img_tensor.to(dtype=torch.get_default_dtype()).div(255)

        return img_tensor

    def _get_image_metainfo(self, index: int, bypass_subset_indices: bool = False) -> dict[str, Any]:
        """Get metadata for image at given index."""
        if not bypass_subset_indices:
            index = self.subset_indices[index]
            
        if self.return_frame_by_frame:
            resource_id, frame_index = self.__find_index(index)
            img_metainfo = dict(self.images_metainfo[resource_id])  # Copy
            img_metainfo['frame_index'] = frame_index
            img_metainfo['annotations'] = [
                ann for ann in img_metainfo['annotations']
                if ann['index'] is None or ann['index'] == frame_index
            ]
        else:
            img_metainfo = self.images_metainfo[index]
            
        return img_metainfo

    def __find_index(self, index: int) -> tuple[int, int]:
        """Find the resource index and frame index for a given global frame index."""
        resource_index = np.searchsorted(self._cumulative_frames[1:], index, side='right')
        frame_index = index - self._cumulative_frames[resource_index]
        return resource_index, frame_index

    def __getitem_internal(
        self, 
        index: int,
        only_load_metainfo: bool = False
    ) -> dict[str, Tensor | FileDataset | dict | list]:
        """Internal method to get item at index."""
        if self.return_frame_by_frame:
            resource_index, frame_idx = self.__find_index(index)
        else:
            resource_index = index
            frame_idx = None
            
        img_metainfo = self._get_image_metainfo(index, bypass_subset_indices=True)

        if only_load_metainfo:
            return {'metainfo': img_metainfo}

        filepath = os.path.join(self.dataset_dir, img_metainfo['file'])
        img, ds = self._load_image(filepath, frame_idx)

        return self._build_item_dict(img, ds, img_metainfo)

    def _build_item_dict(
        self, 
        img: Tensor, 
        ds: FileDataset | None, 
        img_metainfo: dict
    ) -> dict[str, Any]:
        """Build the return dictionary for __getitem__."""
        ret = {'image': img}

        if self.return_dicom:
            ret['dicom'] = ds
        if self.return_metainfo:
            ret['metainfo'] = {k: v for k, v in img_metainfo.items() if k != 'annotations'}
        if self.return_annotations:
            ret['annotations'] = img_metainfo['annotations']

        return ret

    def _filter_annotations(self, annotations: list[dict]) -> list[dict]:
        """Filter annotations based on the filtering settings."""
        if annotations is None:
            return []

        filtered_annotations = []
        for ann in annotations:
            if not self._should_include_annotation(ann):
                continue
            filtered_annotations.append(ann)

        return filtered_annotations

    def _should_include_annotation(self, ann: dict) -> bool:
        """Check if an annotation should be included based on all filters."""
        if not self._should_include_annotator(ann['added_by']):
            return False

        if ann['type'] == 'segmentation':
            return self._should_include_segmentation(ann['name'])
        elif ann['type'] == 'label':
            if ann.get('index', None) is None:
                return self._should_include_image_label(ann['name'])
            else:
                return self._should_include_frame_label(ann['name'])
        
        return True

    def __getitem__(self, index: int) -> dict[str, Tensor | FileDataset | dict | list]:
        """Get item at index.

        Args:
            index: Index

        Returns:
            A dictionary containing 'image', 'metainfo' and 'annotations' keys.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")

        return self.__getitem_internal(self.subset_indices[index])

    def __iter__(self):
        """Iterate over dataset items."""
        for index in self.subset_indices:
            yield self.__getitem_internal(index)

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.subset_indices)

    def _check_version(self) -> None:
        """Check if local dataset version is up to date."""
        metainfo_path = os.path.join(self.dataset_dir, 'dataset.json')
        if not os.path.exists(metainfo_path):
            self.download_project()
            return
            
        with open(metainfo_path, 'r') as file:
            local_dataset_info = json.load(file)
            
        local_updated_at = local_dataset_info.get('updated_at', None)
        local_all_annotations = local_dataset_info.get('all_annotations', None)

        try:
            external_metadata_info = self._get_datasetinfo()
            server_updated_at = external_metadata_info['updated_at']
        except Exception as e:
            _LOGGER.warning(f"Failed to check for updates in {self.project_name}: {e}")
            return

        _LOGGER.debug(f"Local updated at: {local_updated_at}, Server updated at: {server_updated_at}")

        annotations_changed = local_all_annotations != self.all_annotations
        version_outdated = local_updated_at is None or local_updated_at < server_updated_at

        if annotations_changed or version_outdated:
            if annotations_changed:
                _LOGGER.info(
                    f"The 'all_annotations' parameter has changed. "
                    f"Previous: {local_all_annotations}, Current: {self.all_annotations}."
                )
            else:
                _LOGGER.info(
                    f"A newer version of the dataset is available. "
                    f"Your version: {local_updated_at}. Last version: {server_updated_at}."
                )
            self.download_project()
        else:
            _LOGGER.info('Local version is up to date with the latest version.')

    def __add__(self, other):
        """Concatenate datasets."""
        from torch.utils.data import ConcatDataset
        return ConcatDataset([self, other])

    def get_dataloader(self, *args, **kwargs) -> DataLoader:
        """Returns a DataLoader for the dataset with proper collate function.

        Args:
            *args: Positional arguments for the DataLoader.
            **kwargs: Keyword arguments for the DataLoader.

        Returns:
            DataLoader instance with custom collate function.
        """
        return DataLoader(self, *args, collate_fn=self.get_collate_fn(), **kwargs)

    def get_collate_fn(self) -> Callable:
        """Get collate function for DataLoader."""
        def collate_fn(batch: list[dict]) -> dict:
            if not batch:
                return {}
                
            keys = batch[0].keys()
            collated_batch = {}
            
            for key in keys:
                values = [item[key] for item in batch]
                
                if isinstance(values[0], torch.Tensor):
                    shapes = [tensor.shape for tensor in values]
                    if all(shape == shapes[0] for shape in shapes):
                        collated_batch[key] = torch.stack(values)
                    else:
                        _LOGGER.warning(f"Collating {key} tensors with different shapes: {shapes}")
                        collated_batch[key] = values
                elif isinstance(values[0], np.ndarray):
                    collated_batch[key] = np.stack(values)
                else:
                    collated_batch[key] = values

            return collated_batch

        return collate_fn

    def subset(self, indices: list[int]) -> 'DatamintBaseDataset':
        """Create a subset of the dataset.

        Args:
            indices: List of indices to include in the subset.

        Returns:
            Self with updated subset indices.
        """
        if max(indices, default=-1) >= self.dataset_length:
            raise ValueError(f"Subset indices must be less than the dataset length: {self.dataset_length}")

        self.subset_indices = indices
        return self

    def _should_include_annotator(self, annotator_id: str) -> bool:
        """Check if an annotator should be included based on filtering settings."""
        if self.include_annotators is not None:
            return annotator_id in self.include_annotators
        if self.exclude_annotators is not None:
            return annotator_id not in self.exclude_annotators
        return True

    def _should_include_segmentation(self, segmentation_name: str) -> bool:
        """Check if a segmentation should be included based on filtering settings."""
        if self.include_segmentation_names is not None:
            return segmentation_name in self.include_segmentation_names
        if self.exclude_segmentation_names is not None:
            return segmentation_name not in self.exclude_segmentation_names
        return True

    def _should_include_image_label(self, label_name: str) -> bool:
        """Check if an image label should be included based on filtering settings."""
        if self.include_image_label_names is not None:
            return label_name in self.include_image_label_names
        if self.exclude_image_label_names is not None:
            return label_name not in self.exclude_image_label_names
        return True

    def _should_include_frame_label(self, label_name: str) -> bool:
        """Check if a frame label should be included based on filtering settings."""
        if self.include_frame_label_names is not None:
            return label_name in self.include_frame_label_names
        if self.exclude_frame_label_names is not None:
            return label_name not in self.exclude_frame_label_names
        return True
