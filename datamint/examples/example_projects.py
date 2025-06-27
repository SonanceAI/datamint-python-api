import requests
import io
from datamint import APIHandler
import logging
from PIL import Image
import numpy as np

_LOGGER = logging.getLogger(__name__)


def _download_pydicom_test_file(filename: str) -> io.BytesIO:
    """Download a pydicom test file from GitHub and return its content as a BytesIO object."""
    url = f'https://raw.githubusercontent.com/pydicom/pydicom/master/tests/data/{filename}'
    response = requests.get(url)
    response.raise_for_status()
    content = io.BytesIO(response.content)
    content.name = filename
    return content


class ProjectMR:
    @staticmethod
    def upload_resource_emri_small(api: APIHandler = None) -> str:
        if api is None:
            api = APIHandler()

        searched_res = api.get_resources(status='published', tags=['example'], filename='emri_small.dcm')
        for res in searched_res:
            _LOGGER.info('Resource already exists.')
            return res['id']

        dcm_content = _download_pydicom_test_file('emri_small.dcm')

        _LOGGER.info(f'Uploading resource {dcm_content.name}...')
        return api.upload_resources(dcm_content,
                                    anonymize=True,
                                    publish=True,
                                    tags=['example'])

    @staticmethod
    def _upload_annotations(api: APIHandler,
                            resid: str,
                            proj) -> None:
        _LOGGER.info('Uploading annotations...')
        proj_id = proj['id']
        proj_info = api.get_project_by_id(proj_id)
        segurl = 'https://github.com/user-attachments/assets/8c5d7dfe-1b5a-497d-b76e-fe790f09bb90'
        resp = requests.get(segurl, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('L')
        api.upload_segmentations(resid, np.array(img),
                                 name='object1', frame_index=1,
                                 worklist_id=proj_info['worklist_id'])
        api.set_annotation_status(project_id=proj_id,
                                  resource_id=resid,
                                  status='closed')

    @staticmethod
    def create(project_name: str = 'Example Project MR',
               with_annotations=True) -> str:
        api = APIHandler()

        resid = ProjectMR.upload_resource_emri_small(api)
        proj = api.get_project_by_name(project_name)
        if 'id' in proj:
            msg = f'Project {project_name} already exists. Delete it first or choose another name.'
            raise ValueError(msg)
        _LOGGER.info(f'Creating project {project_name}...')
        proj = api.create_project(name=project_name,
                                  description='This is an example project',
                                  resources_ids=[resid])
        if with_annotations:
            ProjectMR._upload_annotations(api, resid, proj)

        return proj['id']
