import os
import requests
from tqdm import tqdm
from requests import Session
from typing import Optional, Callable, Any, Tuple, List, Dict
import logging
import shutil
import json
import yaml
import pydicom
import numpy as np
from datamintapi import configs
from torch.utils.data import DataLoader
import torch
from torchvision.transforms.functional import to_tensor
from pydicom.pixels import pixel_array
from .api_handler import APIHandler
from .base_api_handler import DatamintException
from collections import defaultdict
from PIL import Image

_LOGGER = logging.getLogger(__name__)


class DatamintDatasetException(DatamintException):
    pass


class DatamintDataset:
    """
    Class to download and load datasets from the Datamint API.

    Args:
        root (str): Root directory of dataset where data already exists or will be downloaded.
        project_name (str): Name of the project to download.
        dataset_name (str): Name of the dataset to download. Deprecated, use 'project_name' instead.
        version (int | str): Version of the dataset to download.
            If 'latest', the latest version will be downloaded. Default: 'latest'.
            .. deprecated:: 0.7.0
        auto_update (bool): If True, the dataset will be checked for updates and downloaded if necessary.
        api_key (str, optional): API key to access the Datamint API. If not provided, it will look for the
            environment variable 'DATAMINT_API_KEY'. Not necessary if
            you don't want to download/update the dataset.
        transform (callable, optional): A function/transform that takes in an image (or a series of images)
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the dataset metadata and transforms it.
    """

    def __init__(self,
                 root: str,
                 project_name: str,
                 dataset_name: str = None,
                 version=None,
                 auto_update: bool = True,
                 api_key: Optional[str] = None,
                 image_transform: Callable[[torch.Tensor], Any] = None,
                 mask_transform: Callable[[torch.Tensor], Any] = None,
                 server_url: Optional[str] = None,
                 return_dicom: bool = False,
                 return_metainfo: bool = True,
                 return_seg_annotations: bool = True,
                 return_frame_by_frame: bool = False,
                 return_as_semantic_segmentation: bool = False,
                 ):
        self.api_handler = APIHandler(root_url=server_url, api_key=api_key)
        self.server_url = self.api_handler.root_url
        if isinstance(root, str):
            root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Root directory not found: {root}")

        self.root = root

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.return_dicom = return_dicom
        self.return_metainfo = return_metainfo
        self.return_seg_annotations = return_seg_annotations
        self.return_frame_by_frame = return_frame_by_frame
        self.return_as_semantic_segmentation = return_as_semantic_segmentation

        if version is not None:
            _LOGGER.warning("The 'version' argument is deprecated and will be removed in future versions. " +
                            "See the 'auto_update' argument instead.")

        if version is not None:
            _LOGGER.warning("The 'version' argument is deprecated and will be removed in future versions. " +
                            "See the 'auto_update' argument instead.")

        if dataset_name is not None and project_name is not None:
            raise ValueError("Either 'dataset_name' or 'project_name' must be provided, not both.")

        self.dataset_name = dataset_name
        self.project_name = project_name
        if project_name is not None:
            dataset_name = project_name

        self.dataset_dir = os.path.join(root, dataset_name)
        self.dataset_zippath = os.path.join(root, f'{dataset_name}.zip')

        local_dataset_exists = os.path.exists(os.path.join(self.dataset_dir, 'dataset.json'))

        if project_name is not None and not (local_dataset_exists and auto_update == False):
            self.project_info = self.get_info()
            self.dataset_id = self.project_info['dataset_id']
        else:
            self.dataset_id = None

        self.api_key = self.api_handler.api_key
        if self.api_key is None:
            _LOGGER.warning("API key not provided. If you want to download data, please provide an API key, " +
                            f"eifther by passing it as an argument," +
                            f"setting enviroment variable {configs.ENV_VARS[configs.APIKEY_KEY]} or " +
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
            self.download()

        # Loads the metadata
        if not hasattr(self, 'metainfo'):
            with open(os.path.join(self.dataset_dir, 'dataset.json'), 'r') as file:
                self.metainfo = json.load(file)
        self.images_metainfo = self.metainfo['resources']
        self._check_integrity()

        # fix images_metainfo labels
        for imginfo in self.images_metainfo:
            if imginfo['frame_labels'] is not None:
                for flabels in imginfo['frame_labels']:
                    if flabels['label'] is None:
                        flabels['label'] = []
                    elif isinstance(flabels['label'], str):
                        flabels['label'] = flabels['label'].split(',')

        if self.return_frame_by_frame:
            _LOGGER.debug("Loading frame-by-frame metadata...")
            self.num_frames_per_dicom = []
            for imginfo in self.images_metainfo:
                filepath = os.path.join(self.dataset_dir, imginfo['file'])
                # loads the dicom file
                ds = pydicom.dcmread(filepath)
                self.num_frames_per_dicom.append(ds.NumberOfFrames if hasattr(ds, 'NumberOfFrames') else 1)
            self.dataset_length = sum(self.num_frames_per_dicom)
        else:
            self.dataset_length = len(self.images_metainfo)

        self.subset_indices = list(range(self.dataset_length))
        self.labels_set, self.label2code, self.segmentation_labels, self.segmentation_label2code = self.get_labels_set()
        self.code2label = {v: k for k, v in self.label2code.items()}
        self.num_labels = len(self.labels_set)
        self.num_segmentation_labels = len(self.segmentation_labels)

    def get_resources_ids(self) -> List[str]:
        return [self.__getitem_internal(i, only_load_metainfo=True)['metainfo']['id'] for i in range(len(self))]

    def get_labels_set(self) -> Tuple[List[str], Dict[str, int], List[str], Dict[str, int]]:
        """
        Returns the set of labels and a dictionary that maps labels to integers.

        Returns:
            Tuple[List[str], Dict[str, int]]: The set of labels and the dictionary that maps labels to integers
        """
        if hasattr(self, 'labels_set'):
            return self.labels_set, self.label2code

        all_labels = set()
        segmentation_labels = set()
        for imginfo in self.images_metainfo:
            if 'frame_labels' in imginfo and imginfo['frame_labels'] is not None:
                labels = [l for flabels in imginfo['frame_labels'] for l in flabels['label']]
                all_labels.update(labels)
            if 'annotations' in imginfo and imginfo['annotations'] is not None:
                for ann in imginfo['annotations']:
                    if ann['type'] == 'segmentation':
                        segmentation_labels.update([ann['name']])

        all_labels = sorted(list(all_labels))
        label2code = {label: idx for idx, label in enumerate(all_labels)}
        segmentation_labels = sorted(list(segmentation_labels))
        segmentation_label2code = {label: idx+1 for idx, label in enumerate(segmentation_labels)}
        return all_labels, label2code, segmentation_labels, segmentation_label2code

    def get_framelabel_distribution(self, normalize=False) -> Dict[str, float]:
        """
        Returns the distribution of labels in the dataset.

        Returns:
            Dict[str, int]: The distribution of labels in the dataset.
        """
        label_distribution = {label: 0 for label in self.labels_set}
        for imginfo in self.images_metainfo:
            if 'frame_labels' in imginfo and imginfo['frame_labels'] is not None:
                for flabels in imginfo['frame_labels']:
                    for l in flabels['label']:
                        label_distribution[l] += 1

        if normalize:
            total = sum(label_distribution.values())
            if total == 0:
                return label_distribution
            label_distribution = {k: v/total for k, v in label_distribution.items()}
        return label_distribution

    def get_segmentationlabel_distribution(self, normalize=False) -> Dict[str, float]:
        """
        Returns the distribution of segmentation labels in the dataset.

        Returns:
            Dict[str, int]: The distribution of segmentation labels in the dataset.
        """
        label_distribution = {label: 0 for label in self.segmentation_labels}
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

    def _get_datasetinfo(self) -> Dict:
        # FIXME: use `APIHandler.get_datastsinfo_by_name` instead of direct requests

        request_params = {
            'method': 'GET',
            'url': f'{self.server_url}/datasets',
            'headers': {'apikey': self.api_key}
        }
        with Session() as session:
            response = self._run_request(session, request_params)
            resp = response.json()

        if self.dataset_id is not None:
            value_to_search = self.dataset_id
            field_to_search = 'id'
        else:
            value_to_search = self.dataset_name
            field_to_search = 'name'

        for d in resp['data']:
            if d[field_to_search] == value_to_search:
                return d

        available_datasets = [(d['name'], d['id']) for d in resp['data']]
        raise DatamintDatasetException(
            f"Dataset with {field_to_search} '{value_to_search}' not found." +
            f" Available datasets: {available_datasets}"
        )

    def get_info(self) -> Dict:
        all_projects = self.api_handler.get_projects()
        for project in all_projects:
            if project['name'] == self.project_name:
                return project

        available_projects = [p['name'] for p in all_projects]
        raise DatamintDatasetException(
            f"Project with name '{self.project_name}' not found. Available projects: {available_projects}"
        )

    def _run_request(self, session, request_args) -> requests.Response:
        response = session.request(**request_args)
        response.raise_for_status()
        return response

    def _get_jwttoken(self, dataset_id, session) -> str:
        if dataset_id is None:
            raise ValueError("Dataset ID is required to download the dataset.")
        request_params = {
            'method': 'GET',
            'url': f'{self.server_url}/datasets/{dataset_id}/download/dicom',
            'headers': {'apikey': self.api_key},
            'stream': True
        }
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
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        if self.image_transform is not None:
            body += [repr(self.image_transform)]
        if self.mask_transform is not None:
            body += [repr(self.mask_transform)]
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def download(self):
        """
        Downloads the dataset from the Sonance API into the root directory `self.root`.

        Raises:
            SonanceDatasetException: If the download fails.
        """
        from torchvision.datasets.utils import extract_archive

        dataset_info = self._get_datasetinfo()
        self.dataset_id = dataset_info['id']
        self.last_updaded_at = dataset_info['updated_at']

        with Session() as session:
            jwt_token = self._get_jwttoken(self.dataset_id, session)

            # Initiate the download
            request_params = {
                'method': 'GET',
                'url': f'{self.server_url}/datasets/download/{jwt_token}',
                'headers': {'apikey': self.api_key},
                'stream': True
            }
            response = self._run_request(session, request_params)
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
                with open(self.dataset_zippath, 'wb') as file:
                    for data in response.iter_content(1024):
                        progress_bar.update(len(data))
                        file.write(data)
            if total_size != 0 and progress_bar.n != total_size:
                raise DatamintDatasetException("Download failed.")

            if os.path.exists(self.dataset_dir):
                _LOGGER.info(f"Deleting existing dataset directory: {self.dataset_dir}")
                shutil.rmtree(self.dataset_dir)
            extract_archive(self.dataset_zippath,
                            self.dataset_dir,
                            remove_finished=True
                            )
        with open(os.path.join(self.dataset_dir, 'dataset.json'), 'r') as file:
            self.metainfo = json.load(file)
        if 'updated_at' not in self.metainfo:
            self.metainfo['updated_at'] = self.last_updaded_at
        # save the updated_at date
        with open(os.path.join(self.dataset_dir, 'dataset.json'), 'w') as file:
            json.dump(self.metainfo, file)

    def _load_image(self, filepath: str, index: int = None) -> Tuple[torch.Tensor, pydicom.FileDataset]:
        ds = pydicom.dcmread(filepath)

        if self.return_frame_by_frame:
            img = pixel_array(ds, index=index)
        else:
            img = ds.pixel_array
        # Free up memory
        if hasattr(ds, '_pixel_array'):
            ds._pixel_array = None
        if hasattr(ds, 'PixelData'):
            ds.PixelData = None

        if img.dtype == np.uint16:
            # Pytorch doesn't support uint16
            img = (img//256).astype(np.uint8)

        if len(img.shape) == 3:
            # from (C, H, W) to (H, W, C) because ToTensor() expects the last dimension to be the channel, although it outputs (C, H, W).
            img = img.transpose(1, 2, 0)
        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        img = to_tensor(img)  # Converts to torch.Tensor, normalizes to [0, 1] and changes the shape to (C, H, W)

        return img, ds

    def _process_frame_labels(self, frame_labels: List[dict], img_frame_idx: int, num_frames: int) -> torch.Tensor:
        # Convert frame_labels into a dictionary of int tensors, where the key is the user_id
        # and the value is a tensor of size (num_frames, num_labels)
        # If return_frame_by_frame is True, the size is (num_labels, )
        if self.return_frame_by_frame:
            labels_ret_size = (len(self.labels_set),)
        else:
            labels_ret_size = (num_frames, len(self.labels_set))

        if frame_labels is None:
            return torch.zeros(size=labels_ret_size, dtype=torch.int32)

        frame_labels_byuser = defaultdict(lambda: torch.zeros(size=labels_ret_size, dtype=torch.int32))
        for flabel in frame_labels:
            user_id = flabel['created_by']
            frame_idx = flabel['frame']
            if self.return_frame_by_frame:
                if frame_idx != img_frame_idx:
                    continue
                labels_onehot_i = frame_labels_byuser[user_id]
            else:
                labels_onehot_i = frame_labels_byuser[user_id][frame_idx]

            for l in flabel['label']:
                label_code = self.label2code[l]
                labels_onehot_i[label_code] = 1

        # if no frame labels found, return zeros
        if len(frame_labels_byuser) == 0:
            return torch.zeros(size=labels_ret_size, dtype=torch.int32)

        # merge all user labels using max value
        labels_onehot_merged = torch.stack(list(frame_labels_byuser.values())).max(dim=0)[0]

        return labels_onehot_merged

    def __getitem_internal(self, index: int, only_load_metainfo=False) -> Dict[str, Any]:
        if self.return_frame_by_frame:
            # Find the correct filepath and index
            for i, num_frames in enumerate(self.num_frames_per_dicom):
                if index < num_frames:
                    img_metainfo = self.images_metainfo[i]
                    break
                index -= num_frames

            img_metainfo = dict(img_metainfo)  # copy
            # insert frame index
            img_metainfo['frame_index'] = index
            if self.return_metainfo and self.return_seg_annotations:
                img_metainfo['annotations'] = [ann for ann in img_metainfo['annotations'] if ann['index'] == index]
        else:
            img_metainfo = self.images_metainfo[index]

        # FIXME: deal with multiple annotators

        if self.return_seg_annotations == False:
            img_metainfo.pop('annotations', None)

        if only_load_metainfo:
            return {'metainfo': img_metainfo}

        filepath = os.path.join(self.dataset_dir, img_metainfo['file'])

        # Can be multi-frame, Gray-scale and/or RGB. So the shape is really variable, but it's always a numpy array.
        img, ds = self._load_image(filepath, index)

        segmentations = [None] * img.shape[0]
        seg_labels = [None] * img.shape[0]
        # Load segmentation annotations
        if self.return_seg_annotations:
            for ann in img_metainfo['annotations']:
                if ann['type'] == 'segmentation':
                    if 'file' not in ann:
                        _LOGGER.warning(f"Segmentation annotation without file in {img_metainfo['file']})")
                        continue
                    segfilepath = ann['file']  # png file
                    segfilepath = os.path.join(self.dataset_dir, segfilepath)
                    # FIXME: avoid enforcing resizing the mask
                    seg = np.array(Image.open(segfilepath).convert('L').resize(
                        (img.shape[2], img.shape[1]), Image.NEAREST))
                    seg = torch.from_numpy(seg)
                    seg = seg == 255   # binary mask
                    # map the segmentation label to the code
                    seg_code = self.segmentation_label2code[ann['name']]
                    if self.return_frame_by_frame:
                        frame_index = 0
                    else:
                        frame_index = ann['index']

                    if segmentations[frame_index] is None:
                        segmentations[frame_index] = []
                        seg_labels[frame_index] = []

                    segmentations[frame_index].append(seg)
                    seg_labels[frame_index].append(seg_code)

        # convert to tensor
        for i in range(len(segmentations)):
            if segmentations[i] is not None:
                segmentations[i] = torch.stack(segmentations[i])
                seg_labels[i] = torch.tensor(seg_labels[i], dtype=torch.int32)
            else:
                segmentations[i] = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.bool)
                seg_labels[i] = torch.zeros(1, dtype=torch.int32)

        # process frame_labels
        frame_labels = img_metainfo['frame_labels']
        labels_onehot = self._process_frame_labels(frame_labels, index, img.shape[0])
        # labels_onehot has shape (num_frames, num_labels) or (num_labels, ) if return_frame_by_frame is True

        if self.image_transform is not None:
            img = self.image_transform(img)
        if self.mask_transform is not None and segmentations is not None:
            for i in range(len(segmentations)):
                if segmentations[i] is not None:
                    segmentations[i] = self.mask_transform(segmentations[i])

        if self.return_as_semantic_segmentation:
            if segmentations is not None:
                new_segmentations = torch.zeros((len(segmentations), len(self.segmentation_labels)+1, img.shape[1], img.shape[2]),
                                                dtype=torch.uint8)
                for i in range(len(segmentations)):
                    # for each frame
                    new_segmentations[i, seg_labels[i]] += segmentations[i]
                new_segmentations = new_segmentations > 0
                # pixels that are not in any segmentation are labeled as background
                new_segmentations[:, 0] = new_segmentations.sum(dim=1) == 0
                segmentations = new_segmentations.float()

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        if self.return_frame_by_frame:
            segmentations = segmentations[0]

        ret = {'image': img}

        if self.return_dicom:
            ret['dicom'] = ds
        if self.return_metainfo:
            ret['metainfo'] = img_metainfo
        if self.return_seg_annotations:
            ret['annotations'] = img_metainfo['annotations']
            ret['segmentations'] = segmentations
            ret['seg_labels'] = seg_labels

        ret['labels'] = labels_onehot

        return ret

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, dicom_metadata, metadata) Transformed image, dicom_metadata, and transformed metadata.
                If no transformation is given, the image is a tensor of shape (C, H, W).
        """
        if index < 0 or index >= len(self):
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
            self.download()
            return
        with open(metainfo_path, 'r') as file:
            local_dataset_info = json.load(file)
        local_updated_at = local_dataset_info.get('updated_at', None)

        try:
            external_metadata_info = self._get_datasetinfo()
            server_updated_at = external_metadata_info['updated_at']
        except Exception as e:
            _LOGGER.warning(f"Failed to check for updates in {self.dataset_name}: {e}")
            return

        if local_updated_at is None or local_updated_at != server_updated_at:
            _LOGGER.info(
                f"A newer version of the dataset is available. Your version: {local_updated_at}." +
                f" Last version: {server_updated_at}."
            )
            self.download()
        _LOGGER.info('Local version is up to date with the latest version.')

    def __add__(self, other):
        from torch.utils.data import ConcatDataset
        return ConcatDataset([self, other])

    def get_dataloader(self, *args, batch_size: int, **kwargs) -> DataLoader:
        return DataLoader(self,
                          *args,
                          batch_size=batch_size,
                          collate_fn=self.get_collate_fn(),
                          **kwargs)

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch: Dict) -> Dict:
            keys = batch[0].keys()
            collated_batch = {}
            for key in keys:
                collated_batch[key] = [item[key] for item in batch]
                if isinstance(collated_batch[key][0], torch.Tensor):
                    # check if every tensor has the same shape
                    shapes = [tensor.shape for tensor in collated_batch[key]]
                    if all(shape == shapes[0] for shape in shapes):
                        collated_batch[key] = torch.stack(collated_batch[key])
                elif isinstance(collated_batch[key][0], np.ndarray):
                    collated_batch[key] = np.stack(collated_batch[key])

            return collated_batch

        return collate_fn

    def subset(self, indices: List[int]) -> 'DatamintDataset':
        if len(self.subset_indices) > self.dataset_length:
            raise ValueError(f"Subset indices must be less than the dataset length: {self.dataset_length}")

        self.subset_indices = indices

        return self
