import logging
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

from datamint import configs

_LOGGER = logging.getLogger(__name__)

_DONE_MARKER = '.datamint_download_complete'


def cache_dir(dataset_subdir: str) -> Path:
    if configs.DATAMINT_DATA_DIR is None:
        raise RuntimeError('Could not determine a local data directory to cache the dataset.')
    return Path(configs.DATAMINT_DATA_DIR) / 'examples' / dataset_subdir


def is_cached(dataset_subdir: str) -> bool:
    return (cache_dir(dataset_subdir) / _DONE_MARKER).exists()


def download_and_extract(url: str, dataset_subdir: str) -> Path:
    """Download a zip file from `url` and extract it under the examples cache directory.

    Idempotent: if the dataset was already downloaded and extracted, the cached
    directory is returned without hitting the network again.
    """
    out_dir = cache_dir(dataset_subdir)
    marker = out_dir / _DONE_MARKER
    if marker.exists():
        _LOGGER.info(f'Using cached dataset at {out_dir}')
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / 'download.zip'

    _LOGGER.info(f'Downloading {url}...')
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=dataset_subdir) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    _LOGGER.info(f'Extracting {zip_path}...')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    marker.touch()

    return out_dir
