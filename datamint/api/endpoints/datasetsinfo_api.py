from typing import Literal
from pathlib import Path

from ..entity_base_api import ApiConfig, DeletableEntityApi
from datamint.entities.datasetinfo import DatasetInfo
import httpx
from tqdm.auto import tqdm


class DatasetsInfoApi(DeletableEntityApi[DatasetInfo]):
    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None) -> None:
        """Initialize the datasets API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        super().__init__(config, DatasetInfo, 'datasets', client)

    def get_resources(self,
                      dataset: str | DatasetInfo,
                      version: str | None = None) -> list[dict]:
        """Get all resources belonging to a dataset.

        Args:
            dataset: The dataset ID or DatasetInfo instance.
            version: Optional dataset version string.

        Returns:
            List of resource data dicts.
        """
        dataset_id = self._entid(dataset)
        params: dict = {}
        if version is not None:
            params['version'] = version
        response = self._make_entity_request('GET', dataset_id, add_path='resources', params=params or None)
        return response.json()

    def update_resources(self,
                         dataset: str | DatasetInfo,
                         resource_ids_to_add: list[str] | None = None,
                         resource_ids_to_delete: list[str] | None = None,
                         project_id: str | None = None) -> None:
        """Add or remove resources from a dataset.

        Args:
            dataset: The dataset ID or DatasetInfo instance.
            resource_ids_to_add: List of resource IDs to add.
            resource_ids_to_delete: List of resource IDs to remove.
            project_id: Optional project ID context.
        """
        payload: dict = {'all_files_selected': False}
        if resource_ids_to_add is not None:
            payload['resource_ids_to_add'] = resource_ids_to_add
        if resource_ids_to_delete is not None:
            payload['resource_ids_to_delete'] = resource_ids_to_delete
        if project_id is not None:
            payload['project_id'] = project_id
        dataset_id = self._entid(dataset)
        self._make_entity_request('POST', dataset_id, add_path='resources', json=payload)

    def download(self,
                 dataset: str | DatasetInfo,
                 output_path: str | Path,
                 format: Literal['jpg', 'png', 'tiff', 'dicom', 'npy', 'nifti'] = 'dicom',
                 progress_bar: bool = True) -> None:
        """Download all dataset resources in a given format.

        The server first issues a time-limited download token (``GET
        /datasets/{id}/download/{format}``), then the actual binary archive is
        streamed via that token (``GET /datasets/download/{token}``).

        Args:
            dataset: The dataset ID or DatasetInfo instance.
            output_path: Local path where the downloaded archive will be saved.
            format: Output format for the resources.
            progress_bar: Whether to display a download progress bar.
        """
        dataset_id = self._entid(dataset)
        token_resp = self._make_entity_request('GET', dataset_id, add_path=f'download/{format}')
        token = token_resp.json()
        output_path = Path(output_path)
        with self._stream_request('GET', f'/datasets/download/{token}') as resp:
            total_size = int(resp.headers.get('content-length', 0)) or None
            with tqdm(total=total_size, unit='B', unit_scale=True, disable=not progress_bar) as pbar:
                with open(output_path, 'wb') as f:
                    for chunk in resp.iter_bytes(8192):
                        pbar.update(len(chunk))
                        f.write(chunk)
