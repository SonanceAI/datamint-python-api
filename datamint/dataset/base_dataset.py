import os
import requests
from tqdm import tqdm
from typing import Optional, Callable, Any, Literal
import logging
import shutil
import json
import yaml
import pydicom
import numpy as np
from datamint import configs
from torch.utils.data import DataLoader
import torch
from datamint.apihandler.base_api_handler import DatamintException
from datamint.utils.dicom_utils import is_dicom
import cv2
from datamint.utils.io_utils import read_array_normalized
from deprecated import deprecated
from datetime import datetime

_LOGGER = logging.getLogger(__name__)


class DatamintDatasetException(DatamintException):
    pass


class DatamintBaseDataset:
    """
    Class to download and load datasets from the Datamint API.

    Args:
        root: Root directory of dataset where data already exists or will be downloaded.
        project_name: Name of the project to download.
        auto_update: If True, the dataset will be checked for updates and downloaded if necessary.
        api_key: API key to access the Datamint API. If not provided, it will look for the
            environment variable 'DATAMINT_API_KEY'. Not necessary if
            you don't want to download/update the dataset.
        return_dicom: If True, the DICOM object will be returned, if the image is a DICOM file.
        return_metainfo: If True, the metainfo of the image will be returned.
        return_annotations: If True, the annotations of the image will be returned.
        return_frame_by_frame: If True, each frame of a video/DICOM/3d-image will be returned separately.
        include_unannotated: If True, images without annotations will be included. If False, images without annotations will be discarded. 
        all_annotations: If True, all annotations will be downloaded, including the ones that are not set as closed/done.
        server_url: URL of the Datamint server. If not provided, it will use the default server.
        include_annotators: List of annotators to include. If None, all annotators will be included. See parameter ``exclude_annotators``.
        exclude_annotators: List of annotators to exclude. If None, no annotators will be excluded. See parameter ``include_annotators``.
        include_segmentation_names: List of segmentation names to include. If None, all segmentations will be included.
        exclude_segmentation_names: List of segmentation names to exclude. If None, no segmentations will be excluded.
        include_image_label_names: List of image label names to include. If None, all image labels will be included.
        exclude_image_label_names: List of image label names to exclude. If None, no image labels will be excluded.
        include_frame_label_names: List of frame label names to include. If None, all frame labels will be included.
        exclude_frame_label_names: List of frame label names to exclude. If None, no frame labels will be excluded.

    """

    DATAMINT_DEFAULT_DIR = ".datamint"
    DATAMINT_DATASETS_DIR = "datasets"

    def __init__(self,
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
                 # filtering parameters
                 include_annotators: Optional[list[str]] = None,
                 exclude_annotators: Optional[list[str]] = None,
                 include_segmentation_names: Optional[list[str]] = None,
                 exclude_segmentation_names: Optional[list[str]] = None,
                 include_image_label_names: Optional[list[str]] = None,
                 exclude_image_label_names: Optional[list[str]] = None,
                 include_frame_label_names: Optional[list[str]] = None,
                 exclude_frame_label_names: Optional[list[str]] = None
                 ):
        from datamint.apihandler.api_handler import APIHandler
        
        if project_name is None:
            raise ValueError("project_name is required.")

        self.all_annotations = all_annotations
        self.api_handler = APIHandler(root_url=server_url, api_key=api_key,
                                      check_connection=auto_update)
        self.server_url = self.api_handler.root_url
        if root is None:
            # store them in the home directory
            root = os.path.join(os.path.expanduser("~"),
                                DatamintBaseDataset.DATAMINT_DEFAULT_DIR)
            root = os.path.join(root, DatamintBaseDataset.DATAMINT_DATASETS_DIR)
            if not os.path.exists(root):
                os.makedirs(root)
        elif isinstance(root, str):
            root = os.path.expanduser(root)
            if not os.path.isdir(root):
                raise NotADirectoryError(f"Root directory not found: {root}")

        self.root = root

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

        # Validate filtering parameters
        if include_annotators is not None and exclude_annotators is not None:
            raise ValueError("Cannot set both include_annotators and exclude_annotators at the same time")

        if include_segmentation_names is not None and exclude_segmentation_names is not None:
            raise ValueError("Cannot set both include_segmentation_names and exclude_segmentation_names at the same time")

        if include_image_label_names is not None and exclude_image_label_names is not None:
            raise ValueError("Cannot set both include_image_label_names and exclude_image_label_names at the same time")

        if include_frame_label_names is not None and exclude_frame_label_names is not None:
            raise ValueError("Cannot set both include_frame_label_names and exclude_frame_label_names at the same time")

        self.project_name = project_name
        dataset_name = project_name

        self.dataset_dir = os.path.join(root, dataset_name)
        self.dataset_zippath = os.path.join(root, f'{dataset_name}.zip')

        local_dataset_exists = os.path.exists(os.path.join(self.dataset_dir, 'dataset.json'))

        if local_dataset_exists and auto_update == False:
            # In this case, we don't need to check the API, so we don't need the id.
            self.dataset_id = None
        else:
            self.project_info = self.get_info()
            self.dataset_id = self.project_info['dataset_id']

        self.api_key = self.api_handler.api_key
        if self.api_key is None:
            _LOGGER.warning("API key not provided. If you want to download data, please provide an API key, " +
                            f"either by passing it as an argument," +
                            f"setting environment variable {configs.ENV_VARS[configs.APIKEY_KEY]} or " +
                            "using datamint-config command line tool."
                            )

        # Download/Updates the dataset, if necessary.
        if local_dataset_exists:
            _LOGGER.info(f"Dataset directory already exists: {self.dataset_dir}")
            if auto_update:
                _LOGGER.info("Checking for updates...")
                self._check_version()
        else:
            if self.api_key is None:
                raise DatamintDatasetException("API key is required to download the dataset.")
            _LOGGER.info(f"No data found at {self.dataset_dir}. Downloading...")
            self.download_project()

        # Loads the metadata
        if not hasattr(self, 'metainfo'):
            with open(os.path.join(self.dataset_dir, 'dataset.json'), 'r') as file:
                self.metainfo = json.load(file)
        self.images_metainfo = self.metainfo['resources']

        # filter annotations
        for imginfo in self.images_metainfo:
            imginfo['annotations'] = self._filter_annotations(imginfo['annotations'])

        # filter out images with no annotations.
        if self.discard_without_annotations:
            original_count = len(self.images_metainfo)
            self.images_metainfo = self._filter_items(self.images_metainfo)
            _LOGGER.info(f"Discarded {original_count - len(self.images_metainfo)} images without annotations.")

        self._check_integrity()

        # fix images_metainfo labels
        # TODO: check tags
        # for imginfo in self.images_metainfo:
        #     if imginfo['frame_labels'] is not None:
        #         for flabels in imginfo['frame_labels']:
        #             if flabels['label'] is None:
        #                 flabels['label'] = []
        #             elif isinstance(flabels['label'], str):
        #                 flabels['label'] = flabels['label'].split(',')

        if self.return_frame_by_frame:
            self.dataset_length = 0
            for imginfo in self.images_metainfo:
                filepath = os.path.join(self.dataset_dir, imginfo['file'])
                self.dataset_length += self.read_number_of_frames(filepath)
        else:
            self.dataset_length = len(self.images_metainfo)

        self.num_frames_per_resource = self.__compute_num_frames_per_resource()

        self.subset_indices = list(range(self.dataset_length))
        # self.labels_set, self.label2code, self.segmentation_labels, self.segmentation_label2code = self.get_labels_set()
        self.frame_lsets, self.frame_lcodes = self._get_labels_set(framed=True)
        self.image_lsets, self.image_lcodes = self._get_labels_set(framed=False)
        self.__logged_uint16_conversion = False
        if self.discard_without_annotations and self.return_frame_by_frame:
            # If we are returning frame by frame, we need to filter out frames without segmentations
            self._filter_unannotated()

    def _filter_items(self, images_metainfo: list[dict]) -> list[dict]:
        return [img for img in images_metainfo if len(img.get('annotations', []))]

    def _filter_unannotated(self):
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
        print(f"Filtered dataset: {len(self.subset_indices)} frames with segmentations")

    def __compute_num_frames_per_resource(self) -> list[int]:
        num_frames_per_dicom = []
        for imginfo in self.images_metainfo:
            filepath = os.path.join(self.dataset_dir, imginfo['file'])
            num_frames_per_dicom.append(self.read_number_of_frames(filepath))
        return num_frames_per_dicom

    @property
    def frame_labels_set(self) -> list[str]:
        """
        Returns the set of independent labels in the dataset.
        This is more related to multi-label tasks.
        """
        return self.frame_lsets['multilabel']

    @property
    def frame_categories_set(self) -> list[tuple[str, str]]:
        """
        Returns the set of categories in the dataset.
        This is more related to multi-class tasks.
        """
        return self.frame_lsets['multiclass']

    @property
    def image_labels_set(self) -> list[str]:
        """
        Returns the set of independent labels in the dataset.
        This is more related to multi-label tasks.
        """
        return self.image_lsets['multilabel']

    @property
    def image_categories_set(self) -> list[tuple[str, str]]:
        """
        Returns the set of categories in the dataset.
        This is more related to multi-class tasks.
        """
        return self.image_lsets['multiclass']

    @property
    def segmentation_labels_set(self) -> list[str]:
        """
        Returns the set of segmentation labels in the dataset.
        """
        return self.frame_lsets['segmentation']

    def _get_annotations_internal(self,
                                  annotations: list[dict],
                                  type: Literal['label', 'category', 'segmentation', 'all'] = 'all',
                                  scope: Literal['frame', 'image', 'all'] = 'all') -> list[dict]:
        # check parameters
        if type not in ['label', 'category', 'segmentation', 'all']:
            raise ValueError(f"Invalid value for 'type': {type}")
        if scope not in ['frame', 'image', 'all']:
            raise ValueError(f"Invalid value for 'scope': {scope}")

        annots = []
        for ann in annotations:
            ann_scope = 'image' if ann.get('index', None) is None else 'frame'
            if (type == 'all' or ann['type'] == type) and (scope == 'all' or scope == ann_scope):
                annots.append(ann)
        return annots

    def get_annotations(self,
                        index: int,
                        type: Literal['label', 'category', 'segmentation', 'all'] = 'all',
                        scope: Literal['frame', 'image', 'all'] = 'all') -> list[dict]:
        """
        Returns the annotations of the image at the given index.

        Args:
            index (int): Index of the image.
            type (str): The type of the annotations. It can be 'label', 'category', 'segmentation' or 'all'.
            scope (str): The scope of the annotations. It can be 'frame', 'image' or 'all'.

        Returns:
            List[Dict]: The annotations of the image.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")
        imginfo = self._get_image_metainfo(index)
        return self._get_annotations_internal(imginfo['annotations'], type=type, scope=scope)

    @staticmethod
    def read_number_of_frames(filepath: str) -> int:
        # if is dicom
        if is_dicom(filepath):
            ds = pydicom.dcmread(filepath)
            return ds.NumberOfFrames if hasattr(ds, 'NumberOfFrames') else 1
        # if is a video
        elif filepath.endswith('.mp4') or filepath.endswith('.avi'):
            cap = cv2.VideoCapture(filepath)
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # if is a image
        elif filepath.endswith('.png') or filepath.endswith('.jpg') or filepath.endswith('.jpeg'):
            return 1
        else:
            raise ValueError(f"Unsupported file type: {filepath}")

    def get_resources_ids(self) -> list[str]:
        return [self.__getitem_internal(i, only_load_metainfo=True)['metainfo']['id'] for i in self.subset_indices]

    def _get_labels_set(self, framed: bool) -> tuple[dict, dict[str, dict[str, int]]]:
        """
        Returns the set of labels and a dictionary that maps labels to integers.

        Returns:
            Tuple[List[str], Dict[str, int]]: The set of labels and the dictionary that maps labels to integers
        """

        scope = 'frame' if framed else 'image'

        multilabel_set = set()
        segmentation_labels = set()
        multiclass_set = set()

        for i in range(len(self)):
            anns = self.get_annotations(i, type='label', scope=scope)
            multilabel_set.update([ann['name'] for ann in anns])

            anns = self.get_annotations(i, type='segmentation', scope=scope)
            segmentation_labels.update([ann['name'] for ann in anns])

            anns = self.get_annotations(i, type='category', scope=scope)
            multiclass_set.update([(ann['name'], ann['value']) for ann in anns])

        multilabel_set = sorted(list(multilabel_set))
        multiclass_set = sorted(list(multiclass_set))
        segmentation_labels = sorted(list(segmentation_labels))

        multilabel2code = {label: idx for idx, label in enumerate(multilabel_set)}
        segmentation_label2code = {label: idx+1 for idx, label in enumerate(segmentation_labels)}
        multiclass2code = {label: idx for idx, label in enumerate(multiclass_set)}

        sets = {'multilabel': multilabel_set,
                'segmentation': segmentation_labels,
                'multiclass': multiclass_set}
        codes_map = {'multilabel': multilabel2code,
                     'segmentation': segmentation_label2code,
                     'multiclass': multiclass2code}
        return sets, codes_map

    def get_framelabel_distribution(self, normalize=False) -> dict[str, float]:
        """
        Returns the distribution of labels in the dataset.

        Returns:
            Dict[str, int]: The distribution of labels in the dataset.
        """
        label_distribution = {label: 0 for label in self.frame_labels_set}
        for imginfo in self.images_metainfo:
            for ann in imginfo['annotations']:
                if ann['type'] == 'label' and ann['index'] is not None:
                    label_distribution[ann['name']] += 1

        if normalize:
            total = sum(label_distribution.values())
            if total == 0:
                return label_distribution
            label_distribution = {k: v/total for k, v in label_distribution.items()}
        return label_distribution

    def get_segmentationlabel_distribution(self, normalize=False) -> dict[str, float]:
        """
        Returns the distribution of segmentation labels in the dataset.

        Returns:
            Dict[str, int]: The distribution of segmentation labels in the dataset.
        """
        label_distribution = {label: 0 for label in self.segmentation_labels_set}
        for imginfo in self.images_metainfo:
            if 'annotations' in imginfo and imginfo['annotations'] is not None:
                for ann in imginfo['annotations']:
                    if ann['type'] == 'segmentation':
                        label_distribution[ann['name']] += 1

        if normalize:
            total = sum(label_distribution.values())
            if total == 0:
                return label_distribution
            label_distribution = {k: v/total for k, v in label_distribution.items()}
        return label_distribution

    def _check_integrity(self):
        for imginfo in self.images_metainfo:
            if not os.path.isfile(os.path.join(self.dataset_dir, imginfo['file'])):
                raise DatamintDatasetException(f"Image file {imginfo['file']} not found.")

    def _get_datasetinfo(self) -> dict:
        all_datasets = self.api_handler.get_datasets()

        value_to_search = self.dataset_id
        field_to_search = 'id'

        for d in all_datasets:
            if d[field_to_search] == value_to_search:
                return d

        available_datasets = [(d['name'], d['id']) for d in all_datasets]
        raise DatamintDatasetException(
            f"Dataset with {field_to_search} '{value_to_search}' not found." +
            f" Available datasets: {available_datasets}"
        )

    def get_info(self) -> dict:
        project = self.api_handler.get_project_by_name(self.project_name)
        if 'error' in project:
            available_projects = project['all_projects']
            raise DatamintDatasetException(
                f"Project with name '{self.project_name}' not found. Available projects: {available_projects}"
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
        """
        Example:
            .. code-block:: python

                print(dataset)

            Output:

            .. code-block:: text

                Dataset DatamintDataset
                    Number of datapoints: 3
                    Root location: /home/user/.datamint/datasets

        """
        head = f"Dataset {self.project_name}"
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Location: {self.dataset_dir}")

        # Add filter information to representation
        if self.include_annotators is not None:
            body += [f"Including only annotators: {self.include_annotators}"]
        if self.exclude_annotators is not None:
            body += [f"Excluding annotators: {self.exclude_annotators}"]
        if self.include_segmentation_names is not None:
            body += [f"Including only segmentations: {self.include_segmentation_names}"]
        if self.exclude_segmentation_names is not None:
            body += [f"Excluding segmentations: {self.exclude_segmentation_names}"]
        if self.include_image_label_names is not None:
            body += [f"Including only image labels: {self.include_image_label_names}"]
        if self.exclude_image_label_names is not None:
            body += [f"Excluding image labels: {self.exclude_image_label_names}"]
        if self.include_frame_label_names is not None:
            body += [f"Including only frame labels: {self.include_frame_label_names}"]
        if self.exclude_frame_label_names is not None:
            body += [f"Excluding frame labels: {self.exclude_frame_label_names}"]

        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def download_project(self):
        from torchvision.datasets.utils import extract_archive

        dataset_info = self._get_datasetinfo()
        self.dataset_id = dataset_info['id']
        self.last_updaded_at = dataset_info['updated_at']

        self.api_handler.download_project(self.project_info['id'],
                                          self.dataset_zippath,
                                          all_annotations=self.all_annotations,
                                          include_unannotated=self.include_unannotated)
        _LOGGER.debug(f"Downloaded dataset")
        downloaded_size = os.path.getsize(self.dataset_zippath)
        if downloaded_size == 0:
            raise DatamintDatasetException("Download failed.")

        if os.path.exists(self.dataset_dir):
            _LOGGER.info(f"Deleting existing dataset directory: {self.dataset_dir}")
            shutil.rmtree(self.dataset_dir)
        extract_archive(self.dataset_zippath,
                        self.dataset_dir,
                        remove_finished=True
                        )
        datasetjson = os.path.join(self.dataset_dir, 'dataset.json')
        with open(datasetjson, 'r') as file:
            self.metainfo = json.load(file)
        if 'updated_at' not in self.metainfo:
            self.metainfo['updated_at'] = self.last_updaded_at
        else:
            # if self.last_updated_at is newer than the one in the dataset, update it
            try:
                if datetime.fromisoformat(self.metainfo['updated_at']) < datetime.fromisoformat(self.last_updaded_at):
                    _LOGGER.warning(f"Inconsistent updated_at dates detected ({self.metainfo['updated_at']} < {self.last_updaded_at})." +
                                    f"Fixing it to {self.last_updaded_at}")
                    self.metainfo['updated_at'] = self.last_updaded_at
            except Exception as e:
                _LOGGER.warning(f"Failed to parse updated_at date: {e}")

        # Add all_annotations to the metadata
        self.metainfo['all_annotations'] = self.all_annotations

        # save the updated_at date
        with open(datasetjson, 'w') as file:
            json.dump(self.metainfo, file)

    def _load_image(self, filepath: str, index: int = None) -> tuple[torch.Tensor, pydicom.FileDataset]:
        if os.path.isdir(filepath):
            raise NotImplementedError("Loading a image from a directory is not supported yet.")

        if self.return_frame_by_frame:
            img, ds = read_array_normalized(filepath, return_metainfo=True, index=index)
        else:
            img, ds = read_array_normalized(filepath, return_metainfo=True)

        if img.dtype == np.uint16:
            # Pytorch doesn't support uint16
            if self.__logged_uint16_conversion == False:
                _LOGGER.info("Original image is uint16, converting to uint8")
                self.__logged_uint16_conversion = True

            # min-max normalization
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)

        img = torch.from_numpy(img).contiguous()
        if isinstance(img, torch.ByteTensor):
            img = img.to(dtype=torch.get_default_dtype()).div(255)

        return img, ds

    def _get_image_metainfo(self, index: int, bypass_subset_indices=False) -> dict[str, Any]:
        if bypass_subset_indices == False:
            index = self.subset_indices[index]
        if self.return_frame_by_frame:
            # Find the correct filepath and index
            resource_id, frame_index = self.__find_index(index)

            img_metainfo = self.images_metainfo[resource_id]
            img_metainfo = dict(img_metainfo)  # copy
            # insert frame index
            img_metainfo['frame_index'] = frame_index
            img_metainfo['annotations'] = [ann for ann in img_metainfo['annotations']
                                           if ann['index'] is None or ann['index'] == frame_index]
        else:
            img_metainfo = self.images_metainfo[index]
        return img_metainfo

    def __find_index(self, index: int) -> tuple[int, int]:
        frame_index = index
        for i, num_frames in enumerate(self.num_frames_per_resource):
            if frame_index < num_frames:
                break
            frame_index -= num_frames
        else:
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")

        return i, frame_index

    def __getitem_internal(self, index: int, only_load_metainfo=False) -> dict[str, Any]:
        if self.return_frame_by_frame:
            resource_index, frame_idx = self.__find_index(index)
        else:
            resource_index = index
            frame_idx = None
        img_metainfo = self._get_image_metainfo(index, bypass_subset_indices=True)

        if only_load_metainfo:
            return {'metainfo': img_metainfo}

        filepath = os.path.join(self.dataset_dir, img_metainfo['file'])

        # Can be multi-frame, Gray-scale and/or RGB. So the shape is really variable, but it's always a numpy array.
        img, ds = self._load_image(filepath, frame_idx)

        ret = {'image': img}

        if self.return_dicom:
            ret['dicom'] = ds
        if self.return_metainfo:
            ret['metainfo'] = {k: v for k, v in img_metainfo.items() if k != 'annotations'}
        if self.return_annotations:
            ret['annotations'] = img_metainfo['annotations']

        return ret

    def _filter_annotations(self, annotations: list[dict]) -> list[dict]:
        """
        Filter annotations based on the filtering settings.

        Args:
            annotations: list of annotations

        Returns:
            list[dict]: filtered list of annotations
        """
        if annotations is None:
            return []

        filtered_annotations = []
        for ann in annotations:
            # Filter by annotator
            if not self._should_include_annotator(ann['added_by']):
                continue

            # Filter by annotation type and name
            if ann['type'] == 'segmentation':
                if not self._should_include_segmentation(ann['name']):
                    continue
            elif ann['type'] == 'label':
                # Check if it's a frame or image label
                if ann.get('index', None) is None:
                    # Image label
                    if not self._should_include_image_label(ann['name']):
                        continue
                else:
                    # Frame label
                    if not self._should_include_frame_label(ann['name']):
                        continue

            # If we reach here, the annotation passed all filters
            filtered_annotations.append(ann)

        return filtered_annotations

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            dict: A dictionary containing three keys: 'image', 'metainfo' and 'annotations'.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")

        return self.__getitem_internal(self.subset_indices[index])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self.subset_indices)

    def _check_version(self):
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

        # Check if all_annotations changed or doesn't exist
        annotations_changed = local_all_annotations != self.all_annotations

        if local_updated_at is None or local_updated_at < server_updated_at or annotations_changed:
            if annotations_changed:
                _LOGGER.info(
                    f"The 'all_annotations' parameter has changed. Previous: {local_all_annotations}, Current: {self.all_annotations}."
                )
            else:
                _LOGGER.info(
                    f"A newer version of the dataset is available. Your version: {local_updated_at}." +
                    f" Last version: {server_updated_at}."
                )
            self.download_project()
        else:
            _LOGGER.info('Local version is up to date with the latest version.')

    def __add__(self, other):
        from torch.utils.data import ConcatDataset
        return ConcatDataset([self, other])

    def get_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Returns a DataLoader for the dataset.
        This is a wrapper around the PyTorch DataLoader, with the convinience of using a nice collate_fn
        that properly handles the different types of data in this dataset.

        Args:
            *args: Positional arguments for the DataLoader. See `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ for details.
            **kwargs: Keyword arguments for the DataLoader. See `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ for details.


        """
        return DataLoader(self,
                          *args,
                          collate_fn=self.get_collate_fn(),
                          **kwargs)

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch: dict) -> dict:
            keys = batch[0].keys()
            collated_batch = {}
            for key in keys:
                collated_batch[key] = [item[key] for item in batch]
                if isinstance(collated_batch[key][0], torch.Tensor):
                    # check if every tensor has the same shape
                    shapes = [tensor.shape for tensor in collated_batch[key]]
                    if all(shape == shapes[0] for shape in shapes):
                        collated_batch[key] = torch.stack(collated_batch[key])
                    else:
                        _LOGGER.warning(f"Collating {key} tensors with different shapes: {shapes}. ")
                elif isinstance(collated_batch[key][0], np.ndarray):
                    collated_batch[key] = np.stack(collated_batch[key])

            return collated_batch

        return collate_fn

    def subset(self, indices: list[int]) -> 'DatamintBaseDataset':
        if len(self.subset_indices) > self.dataset_length:
            raise ValueError(f"Subset indices must be less than the dataset length: {self.dataset_length}")

        self.subset_indices = indices

        return self

    def _should_include_annotator(self, annotator_id: str) -> bool:
        """
        Check if an annotator should be included based on the filtering settings.

        Args:
            annotator_id: The ID of the annotator to check

        Returns:
            bool: True if the annotator should be included, False otherwise
        """
        if self.include_annotators is not None:
            return annotator_id in self.include_annotators
        if self.exclude_annotators is not None:
            return annotator_id not in self.exclude_annotators
        return True

    def _should_include_segmentation(self, segmentation_name: str) -> bool:
        """
        Check if a segmentation should be included based on the filtering settings.

        Args:
            segmentation_name: The name of the segmentation to check

        Returns:
            bool: True if the segmentation should be included, False otherwise
        """
        if self.include_segmentation_names is not None:
            return segmentation_name in self.include_segmentation_names
        if self.exclude_segmentation_names is not None:
            return segmentation_name not in self.exclude_segmentation_names
        return True

    def _should_include_image_label(self, label_name: str) -> bool:
        """
        Check if an image label should be included based on the filtering settings.

        Args:
            label_name: The name of the image label to check

        Returns:
            bool: True if the image label should be included, False otherwise
        """
        if self.include_image_label_names is not None:
            return label_name in self.include_image_label_names
        if self.exclude_image_label_names is not None:
            return label_name not in self.exclude_image_label_names
        return True

    def _should_include_frame_label(self, label_name: str) -> bool:
        """
        Check if a frame label should be included based on the filtering settings.

        Args:
            label_name: The name of the frame label to check

        Returns:
            bool: True if the frame label should be included, False otherwise
        """
        if self.include_frame_label_names is not None:
            return label_name in self.include_frame_label_names
        if self.exclude_frame_label_names is not None:
            return label_name not in self.exclude_frame_label_names
        return True
