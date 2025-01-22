import requests
import io
from datamintapi import APIHandler
import logging

_LOGGER = logging.getLogger(__name__)


class ProjectMR:
    @staticmethod
    def upload_resource_emri_small(api: APIHandler = None) -> str:
        if api is None:
            api = APIHandler()

        searched_res = api.get_resources(status='published', tags=['example'], filename='emri_small.dcm')
        for res in searched_res:
            _LOGGER.info('Resource already exists.')
            return res['id']

        download_url = 'https://github.com/robyoung/dicom-test-files/raw/refs/heads/master/data/pydicom/emri_small.dcm'
        response = requests.get(download_url)
        dcm_content = io.BytesIO(response.content)
        dcm_content.name = 'emri_small.dcm'

        _LOGGER.info(f'Uploading resource {dcm_content.name}...')
        return api.upload_resources(dcm_content,
                                    anonymize=True,
                                    publish=True,
                                    tags=['example'])

    @staticmethod
    def create(project_name: str = 'Example Project MR') -> str:
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

        return proj['id']
