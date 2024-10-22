import os
import requests
from tqdm import tqdm
from requests import Session
from typing import Optional, Callable, Any, Tuple
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

_LOGGER = logging.getLogger(__name__)


class DatamintDatasetException(Exception):
    pass


class DatamintDataset:
    """
    Class to download and load datasets from the Datamint API.

    Args:
        root (str): Root directory of dataset where data already exists or will be downloaded.
        dataset_name (str): Name of the dataset to download.
        version (int | str): Version of the dataset to download.
            If 'latest', the latest version will be downloaded. Default: 'latest'.
        api_key (str, optional): API key to access the Datamint API. If not provided, it will look for the
            environment variable 'DATAMINT_API_KEY'. Not necessary if
            you don't want to download/update the dataset.
        transform (callable, optional): A function/transform that takes in an image (or a series of images)
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the dataset metadata and transforms it.
    """

    default_root_url = 'https://api.datamint.io'

    def __init__(self,
                 root: str,
                 dataset_name: str,
                 version: int | str = 'latest',
                 api_key: Optional[str] = None,
                 transform: Callable[[torch.Tensor], Any] = None,
                 target_transform: Callable[[dict], Any] = None,
                 server_url: Optional[str] = None,
                 return_dicom: bool = False,
                 return_metainfo: bool = True,
                 return_frame_by_frame: bool = False,
                 # dicom_transform: Optional[Callable[[pydicom.Dataset], Any]] = None # TODO: Discuss if this will be useful?
                 ):
        if server_url is None:
            _LOGGER.debug(f"Using default server URL: {DatamintDataset.default_root_url}")
            server_url = DatamintDataset.default_root_url
        self.server_url = server_url
        if isinstance(root, str):
            root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Root directory not found: {root}")

        self.root = root

        self.transform = transform
        self.target_transform = target_transform

        self.return_dicom = return_dicom
        self.return_metainfo = return_metainfo
        self.return_frame_by_frame = return_frame_by_frame

        if isinstance(version, str) and version != 'latest':
            raise ValueError("Version must be an integer or 'latest'")

        self.version = version
        self.dataset_name = dataset_name
        self.api_key = api_key if api_key is not None else configs.get_value(configs.APIKEY_KEY)
        if self.api_key is None:
            _LOGGER.warning("API key not provided. If you want to download data, please provide an API key, " +
                            f"either by passing it as an argument," +
                            f"setting enviroment variable {configs.ENV_VARS[configs.APIKEY_KEY]} or " +
                            "using datamint-config command line tool."
                            )
        self.dataset_dir = os.path.join(root, dataset_name)
        self.dataset_zippath = os.path.join(root, f'{dataset_name}.zip')

        # Download/Updates the dataset, if necessary.
        if os.path.exists(self.dataset_dir):
            _LOGGER.info(f"Dataset directory already exists: {self.dataset_dir}")
            _LOGGER.info("Checking for updates...")
            self._check_version()
        else:
            if self.api_key is None:
                raise DatamintDatasetException("API key is required to download the dataset.")
            _LOGGER.info(f"No data found at {self.dataset_dir}. Downloading...")
            self.download()

        # Loads the metadata
        with open(os.path.join(self.dataset_dir, 'dataset.json'), 'r') as file:
            self.metainfo = json.load(file)
        self.images_metainfo = self.metainfo['resources']
        self._check_integrity()

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

    def _check_integrity(self):
        for imginfo in self.images_metainfo:
            if not os.path.isfile(os.path.join(self.dataset_dir, imginfo['file'])):
                raise DatamintDatasetException(f"Image file {imginfo['file']} not found.")

    def _get_datasetinfo_by_name(self, dataset_name: str) -> dict:
        # FIXME: use `APIHandler.get_datastsinfo_by_name` instead of direct requests
        request_params = {
            'method': 'GET',
            'url': f'{self.server_url}/datasets',
            'headers': {'apikey': self.api_key}
        }
        with Session() as session:
            response = self._run_request(session, request_params)
            resp = response.json()
        for d in resp['data']:
            if d['name'] == dataset_name:
                return d

        available_datasets = [d['name'] for d in resp['data']]
        raise DatamintDatasetException(
            f"Dataset with name '{dataset_name}' not found. Available datasets: {available_datasets}"
        )

    def _run_request(self, session, request_args) -> requests.Response:
        response = session.request(**request_args)
        response.raise_for_status()
        return response

    def _get_jwttoken(self, dataset_id, session) -> str:
        request_params = {
            'method': 'GET',
            'url': f'{self.server_url}/datasets/{dataset_id}/download/dicom',
            'headers': {'apikey': self.api_key},
            'params': {'version': self.version},
            'stream': True
        }
        response = self._run_request(session, request_params)
        progress_bar = None
        number_processed_images = 0

        try:
            response_iterator = response.iter_lines(decode_unicode=True)
            for line in response_iterator:
                line = line.strip()
                if 'event: error' in line:
                    error_msg = '\n'.join(response_iterator)
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
        if self.transform is not None:
            body += [repr(self.transforms)]
        if self.target_transform is not None:
            body += [repr(self.target_transform)]
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def download(self):
        """
        Downloads the dataset from the Sonance API into the root directory `self.root`.

        Raises:
            SonanceDatasetException: If the download fails.
        """
        from torchvision.datasets.utils import extract_archive

        dataset_info = self._get_datasetinfo_by_name(self.dataset_name)
        dataset_id = dataset_info['id']
        if self.version == 'latest':
            self.version = dataset_info['updated_at']  # dataset_info['last_version']

        with Session() as session:
            jwt_token = self._get_jwttoken(dataset_id, session)

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

    def __getitem__(self, index: int) -> tuple[Any, pydicom.FileDataset, dict]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, dicom_metadata, metadata) Transformed image, dicom_metadata, and transformed metadata.
        """
        if index < 0 or index >= self.dataset_length:
            raise IndexError(f"Index {index} out of bounds for dataset of length {self.dataset_length}")

        # Find the correct filepath and index
        if self.return_frame_by_frame:
            for num_frames in self.num_frames_per_dicom:
                if index < num_frames:
                    img_metainfo = self.images_metainfo[index]
                    break
                index -= num_frames
        else:
            img_metainfo = self.images_metainfo[index]
        filepath = os.path.join(self.dataset_dir, img_metainfo['file'])

        # Can be multi-frame, Gray-scale and/or RGB. So the shape is really variable, but it's always a numpy array.
        img, ds = self._load_image(filepath, index)

        if self.transform is not None:
            img = self.transform(img)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        if self.target_transform is not None:
            img_metainfo = self.target_transform(img_metainfo)

        ret = [img]
        if self.return_dicom:
            ret.append(ds)
        if self.return_metainfo:
            ret.append(img_metainfo)
        ret = tuple(ret)

        return ret

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return self.dataset_length

    def _check_version(self):
        with open(os.path.join(self.dataset_dir, 'dataset.json'), 'r') as file:
            local_dataset_info = json.load(file)
        if 'version' in local_dataset_info:
            local_version = int(local_dataset_info['version'])
        else:
            _LOGGER.warning("Local version not found in 'dataset.json'. Assuming version 1.")
            local_version = 1
        if isinstance(self.version, int):
            if local_version != self.version:
                self.download()
            return

        try:
            external_metadata_info = self._get_datasetinfo_by_name(self.dataset_name)
            last_version = external_metadata_info['last_version']
            if last_version is None:
                last_version = 1
        except Exception as e:
            _LOGGER.warning(f"Failed to check for updates in {self.dataset_name}: {e}")
            return

        if local_version != last_version:
            print(
                f"A newer version of the dataset is available. Your version: {local_version}." +
                f" Last version: {last_version}.\n Would you like to update?\n (y/n)"
            )
            choice = input().lower()
            if choice == 'y':
                self.download()
            else:
                return
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
        def collate_fn(batch) -> Tuple:
            k = 0
            images = [item[0] for item in batch]
            if isinstance(images[0], torch.Tensor):
                images = torch.stack(images)
            elif isinstance(images[0], np.ndarray):
                images = np.stack(images)
            else:
                _LOGGER.warning(f"Unknown image type: {type(images[0])}. Proceeding with the original structure.")

            collated_batch = [images]
            if self.return_dicom:
                k += 1
                dicom_metainfo = [item[k] for item in batch]
                collated_batch.append(dicom_metainfo)
            if self.return_metainfo:
                k += 1
                metainfo = [item[k] for item in batch]
                collated_batch.append(metainfo)

            return tuple(collated_batch)

        return collate_fn
