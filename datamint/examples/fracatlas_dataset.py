import logging

import requests
from tqdm.auto import tqdm

from datamint import Api, configs
from datamint.entities import Project

from . import _common, _download

_LOGGER = logging.getLogger(__name__)

_FIGSHARE_ARTICLE_URL = 'https://api.figshare.com/v2/articles/22363012'
_DATASET_NAME = 'FracAtlas Classification Example'
_DESCRIPTION = ('Fracture classification example dataset (FracAtlas), '
               'auto-populated by datamint.examples.fracatlas_dataset.')
_LABEL_IDENTIFIER = 'has_fracture'


def _get_download_url() -> str:
    resp = requests.get(_FIGSHARE_ARTICLE_URL)
    resp.raise_for_status()
    return resp.json()['files'][0]['download_url']


def create(project_name: str = _DATASET_NAME, api: Api | None = None) -> Project:
    """Download the FracAtlas fracture-classification dataset and upload it as a Datamint project.

    Source: https://doi.org/10.6084/m9.figshare.22363012.
    """
    api = api or Api()

    proj, already_existed = _common.get_or_create_project(project_name, _DESCRIPTION, api)
    if already_existed:
        _LOGGER.warning(f"Project '{project_name}' already exists. Skipping data population.")
        _common.print_skip_summary(_DATASET_NAME, proj)
        return proj

    if not _download.is_cached('fracatlas'):
        print('This download may take a few minutes.')

    download_url = _get_download_url()
    data_dir = _download.download_and_extract(download_url, 'fracatlas')

    fractured_dir = next(data_dir.rglob('Fractured'))
    non_fractured_dir = next(data_dir.rglob('Non_fractured'))

    fractured_paths = sorted(p for p in fractured_dir.iterdir() if p.is_file())
    non_fractured_paths = sorted(p for p in non_fractured_dir.iterdir() if p.is_file())

    non_fractured_ids = api.resources.upload_resources(
        [str(p) for p in non_fractured_paths],
        tags=['fracatlas', 'non-fractured'],
        publish_to=proj,
        progress_bar=True,
    )
    fractured_ids = api.resources.upload_resources(
        [str(p) for p in fractured_paths],
        tags=['fracatlas', 'fractured'],
        publish_to=proj,
        progress_bar=True,
    )
    # Annotations count toward a project only through one of its worklists,
    # and annotators only see the categories listed in its annotations.
    worklist_id = api.annotationworklists.create(configs.DEFAULT_UPLOAD_WORKLIST_NAME,
                                                 resource_ids=list(non_fractured_ids) + list(fractured_ids),
                                                 project=proj,
                                                 annotations=[{'type': 'category', 'identifier': _LABEL_IDENTIFIER,
                                                               'scope': 'image', 'required': False,
                                                               'values': ['no', 'yes']}],
                                                 return_entity=False)

    for resource_id in tqdm(non_fractured_ids, desc='Uploading annotations'):
        api.annotations.add_category_annotation(
            resource=resource_id,
            identifier=_LABEL_IDENTIFIER,
            value='no',
            worklist_id=worklist_id,
        )
    for resource_id in tqdm(fractured_ids, desc='Uploading annotations'):
        api.annotations.add_category_annotation(
            resource=resource_id,
            identifier=_LABEL_IDENTIFIER,
            value='yes',
            worklist_id=worklist_id,
        )

    n_files = len(non_fractured_ids) + len(fractured_ids)
    _common.print_summary(_DATASET_NAME, n_files, n_files, data_dir, proj)
    return proj
