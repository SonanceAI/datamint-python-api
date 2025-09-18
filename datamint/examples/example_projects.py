import requests
import io
from datamint import Api
import logging
from PIL import Image
import numpy as np
from datamint.entities import Project, Resource
from pydicom.data import get_testdata_file

_LOGGER = logging.getLogger(__name__)


class ProjectMR:
    @staticmethod
    def upload_resource_emri_small(api: Api | None = None) -> Resource:
        if api is None:
            api = Api()

        searched_res = api.resources.get_list(status='published',
                                              tags=['example'],
                                              filename='emri_small.dcm')
        for res in searched_res:
            _LOGGER.info('Resource already exists.')
            return res

        dcm_path = get_testdata_file("emri_small.dcm",
                                     read=False)

        _LOGGER.info('Uploading resource emri_small.dcm...')
        resid = api.resources.upload_resource(dcm_path,
                                              anonymize=False,
                                              publish=True,
                                              tags=['example'])
        return api.resources.get_by_id(resid)

    @staticmethod
    def _upload_annotations(api: Api,
                            res: Resource,
                            proj: Project) -> None:
        _LOGGER.info('Uploading annotations...')
        segurl = 'https://github.com/user-attachments/assets/8c5d7dfe-1b5a-497d-b76e-fe790f09bb90'
        resp = requests.get(segurl, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('L')
        api.annotations.upload_segmentations(res, np.array(img),
                                             name='object1', frame_index=1,
                                             worklist_id=proj.worklist_id)
        api.projects.set_work_status(resource=res,
                                     project=proj,
                                     status='closed')

    @staticmethod
    def create(project_name: str = 'Example Project MR',
               with_annotations=True) -> Project:
        api = Api()

        res = ProjectMR.upload_resource_emri_small(api)
        proj = api.projects.get_by_name(name=project_name)
        if proj:
            _LOGGER.warning(f'Project {project_name} already exists. Returning it without modifications...')
            return proj

        _LOGGER.info(f'Creating project {project_name}...')
        projid = api.projects.create(name=project_name,
                                     description='This is an example project',
                                     resources_ids=[res.id])
        proj = api.projects.get_by_id(projid)

        if with_annotations:
            ProjectMR._upload_annotations(api, res, proj)

        return proj
