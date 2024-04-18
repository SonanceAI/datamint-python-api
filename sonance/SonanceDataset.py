import os
import requests
from tqdm import tqdm
from requests import Session
from typing import Optional, Callable, Any
from torchvision.datasets.utils import extract_archive
import logging
import shutil
import json
import yaml
import pydicom
import numpy as np
from torch.utils.data import Dataset

_LOGGER = logging.getLogger(__name__)


class SonanceDatasetException(Exception):
    pass


class SonanceDataset(Dataset):
    """
    Class to download and load datasets from the Sonance API.

    Args:
        root (str): Root directory of dataset where data already exists or will be downloaded.
        dataset_name (str): Name of the dataset to download.
        version (int | str): Version of the dataset to download.
            If 'latest', the latest version will be downloaded. Default: 'latest'.
        api_key (str, optional): API key to access the Sonance API. If not provided, it will look for the
            environment variable 'SONANCE_DATASET_API_KEY'. Not necessary if
            you dont want to download/update the dataset.
        transform (callable, optional): A function/transform that takes in an image (or a series of images)
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the dataset metadata and transforms it.
    """

    root_url = 'https://stagingapi.datamint.io'

    def __init__(self,
                 root: str,
                 dataset_name: str,
                 version: int | str = 'latest',
                 api_key: Optional[str] = None,
                 transform: Callable[[np.ndarray], Any] = None,
                 target_transform: Callable[[dict], Any] = None
                 # dicom_transform: Optional[Callable[[pydicom.Dataset], Any]] = None # TODO: Discuss if this will be useful?
                 ):
        if isinstance(root, str):
            root = os.path.expanduser(root)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Root directory not found: {root}")

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if isinstance(version, str) and version != 'latest':
            raise ValueError("Version must be an integer or 'latest'")

        self.version = version
        self.dataset_name = dataset_name
        self.api_key = api_key if api_key is not None else os.getenv('SONANCE_DATASET_API_KEY')
        if self.api_key is None:
            _LOGGER.warning("API key not provided. If you want to download, please provide an API key.")
        self.dataset_dir = os.path.join(root, dataset_name)
        self.dataset_zippath = os.path.join(root, f'{dataset_name}.zip')

        # Download/Updates the dataset, if necessary.
        if os.path.exists(self.dataset_dir):
            _LOGGER.info(f"Dataset directory already exists: {self.dataset_dir}")
            _LOGGER.info("Checking for updates...")
            self._check_version()
        else:
            if api_key is None:
                raise SonanceDatasetException("API key is required to download the dataset.")
            _LOGGER.info(f"No data found at {self.dataset_dir}. Downloading...")
            self.download()

        # Loads the metadata
        with open(os.path.join(self.dataset_dir, 'dataset.json'), 'r') as file:
            self.metainfo = json.load(file)
        self.images_metainfo = self.metainfo['images']
        self._check_integrity()

    def _check_integrity(self):
        for imginfo in self.images_metainfo:
            if not os.path.isfile(os.path.join(self.dataset_dir, imginfo['image_file'])):
                raise SonanceDatasetException(f"Image file {imginfo['image_file']} not found.")

    def _get_datasetinfo_by_name(self, dataset_name: str) -> dict:
        request_params = {
            'method': 'GET',
            'url': f'{SonanceDataset.root_url}/datasets',
            'headers': {'apikey': self.api_key}
        }
        with Session() as session:
            response = self._run_request(session, request_params)
            resp = response.json()
        for d in resp['data']:
            if d['name'] == dataset_name:
                return d

        available_datasets = [d['name'] for d in resp['data']]
        raise SonanceDatasetException(
            f"Dataset with name '{dataset_name}' not found. Available datasets: {available_datasets}"
        )

    def _run_request(self, session, request_args) -> requests.Response:
        response = session.request(**request_args)
        response.raise_for_status()
        return response

    def _get_jwttoken(self, dataset_id, session) -> str:
        request_params = {
            'method': 'GET',
            'url': f'{SonanceDataset.root_url}/datasets/{dataset_id}/download/dicom',
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
                    raise SonanceDatasetException(f"Getting jwt token failed:\n{error_msg}")
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

        raise SonanceDatasetException("Getting jwt token failed! No dataline with 'zip' entry found.")

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
        dataset_info = self._get_datasetinfo_by_name(self.dataset_name)
        dataset_id = dataset_info['id']
        if self.version == 'latest':
            self.version = dataset_info['last_version']

        with Session() as session:
            jwt_token = self._get_jwttoken(dataset_id, session)

            # Initiate the download
            request_params = {
                'method': 'GET',
                'url': f'{SonanceDataset.root_url}/datasets/download/{jwt_token}',
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
                raise SonanceDatasetException("Download failed.")

            if os.path.exists(self.dataset_dir):
                _LOGGER.info(f"Deleting existing dataset directory: {self.dataset_dir}")
                shutil.rmtree(self.dataset_dir)
            extract_archive(self.dataset_zippath,
                            self.dataset_dir,
                            remove_finished=True
                            )

    def __getitem__(self, index: int) -> tuple[Any, pydicom.FileDataset, dict]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, dicom_metadata, metadata) Transformed image, dicom_metadata, and transformed metadata.
        """

        img_metainfo = self.images_metainfo[index]

        filepath = os.path.join(self.dataset_dir, img_metainfo['image_file'])
        # loads the dicom file
        ds = pydicom.dcmread(filepath)

        # Can be multi-frame, Gray-scale and/or RGB. So the shape is really variable, but it's always a numpy array.
        img = ds.pixel_array
        if img.dtype == np.uint16:
            # Pytorch doesn't support uint16
            img = (img//256).astype(np.uint8)
        if hasattr(ds, '_pixel_array'):
            ds._pixel_array = None  # Free up memory
        else:
            _LOGGER.warning("ds._pixel_array not found. This may cause memory issues. Check pydicom version.")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            img_metainfo = self.target_transform(img_metainfo)

        return img, ds, img_metainfo

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self.images_metainfo)

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
                f"A newer version of the dataset is available. Your version: {local_version}. Last version: {last_version}.\n Would you like to update?\n (y/n)"
            )
            choice = input().lower()
            if choice == 'y':
                self.download()
            else:
                return
        _LOGGER.info('Local version is up to date with the latest version.')


if __name__ == '__main__':
    # Example usage for testing purposes.
    logging.basicConfig(level=logging.INFO)
    dataset = SonanceDataset(root='../data',
                             dataset_name='TestCTdataset',
                             version='latest')
    print(dataset)
    img, ds, metadata = dataset[0]
    print('Image shape:', img.shape)  # image(s)
    print('Metadata:', metadata)  # metadata
