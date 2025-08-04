from typing import Optional, IO, Sequence, Literal, Generator, Union, Any
import os
import io
import pydicom.dataset
from requests.exceptions import HTTPError
import logging
import asyncio
import aiohttp
from medimgkit.dicom_utils import anonymize_dicom, to_bytesio, is_dicom
from medimgkit import dicom_utils
import pydicom
from pathlib import Path
from datetime import date
import mimetypes
from PIL import Image
import cv2
from nibabel.filebasedimages import FileBasedImage as nib_FileBasedImage
from datamint import configs
from .base_api_handler import BaseAPIHandler, DatamintException, ResourceNotFoundError, ResourceFields, ResourceStatus
from deprecated.sphinx import deprecated
import json
import itertools
from tqdm.auto import tqdm

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')


def _is_io_object(obj):
    """
    Check if an object is a file-like object.
    """
    return callable(getattr(obj, "read", None))


def _infinite_gen(x):
    while True:
        yield x


def _open_io(file_path: str | Path | IO, mode: str = 'rb') -> IO:
    if isinstance(file_path, str) or isinstance(file_path, Path):
        return open(file_path, 'rb')
    return file_path


class RootAPIHandler(BaseAPIHandler):
    """
    Class to handle the API requests to the Datamint API
    """
    DATAMINT_API_VENV_NAME = configs.ENV_VARS[configs.APIKEY_KEY]
    ENDPOINT_RESOURCES = 'resources'
    ENDPOINT_CHANNELS = f'{ENDPOINT_RESOURCES}/channels'

    async def _upload_single_resource_async(self,
                                            file_path: str | IO,
                                            mimetype: Optional[str] = None,
                                            anonymize: bool = False,
                                            anonymize_retain_codes: Sequence[tuple] = [],
                                            tags: list[str] = [],
                                            mung_filename: Sequence[int] | Literal['all'] | None = None,
                                            channel: Optional[str] = None,
                                            session=None,
                                            modality: Optional[str] = None,
                                            publish: bool = False,
                                            metadata_file: Optional[str | dict] = None,
                                            ) -> str:
        if _is_io_object(file_path):
            name = file_path.name
        else:
            name = file_path

        if session is not None and not isinstance(session, aiohttp.ClientSession):
            raise ValueError("session must be an aiohttp.ClientSession object.")

        name = os.path.expanduser(os.path.normpath(name))
        if len(Path(name).parts) == 0:
            raise ValueError(f"File path '{name}' is not valid.")
        name = os.path.join(*[x if x != '..' else '_' for x in Path(name).parts])

        if mung_filename is not None:
            file_parts = Path(name).parts
            if file_parts[0] == os.path.sep:
                file_parts = file_parts[1:]
            if mung_filename == 'all':
                new_file_path = '_'.join(file_parts)
            else:
                folder_parts = file_parts[:-1]
                new_file_path = '_'.join([folder_parts[i-1] for i in mung_filename if i <= len(folder_parts)])
                new_file_path += '_' + file_parts[-1]
            name = new_file_path
            _LOGGER.debug(f"New file path: {name}")

        if mimetype is None:
            mimetype = mimetypes.guess_type(name)[0]
        is_a_dicom_file = None
        if mimetype is None:
            is_a_dicom_file = is_dicom(name) or is_dicom(file_path)
            if is_a_dicom_file:
                mimetype = 'application/dicom'
            elif name.endswith('.nii') or name.endswith('.nii.gz'):
                mimetype = 'application/x-nifti'

        filename = os.path.basename(name)
        _LOGGER.debug(f"File name '{filename}' mimetype: {mimetype}")

        if anonymize:
            if is_a_dicom_file == True or is_dicom(file_path):
                ds = pydicom.dcmread(file_path)
                _LOGGER.info(f"Anonymizing {file_path}")
                ds = anonymize_dicom(ds, retain_codes=anonymize_retain_codes)
                # make the dicom `ds` object a file-like object in order to avoid unnecessary disk writes
                f = to_bytesio(ds, name)
            else:
                _LOGGER.warning(f"File {file_path} is not a dicom file. Skipping anonymization.")
                f = _open_io(file_path)
        else:
            f = _open_io(file_path)

        try:
            metadata_content = None
            metadata_dict = None
            if metadata_file is not None:
                if isinstance(metadata_file, dict):
                    # Metadata is already a dictionary
                    metadata_dict = metadata_file
                    metadata_content = json.dumps(metadata_dict)
                    _LOGGER.debug("Using provided metadata dictionary")
                else:
                    # Metadata is a file path
                    try:
                        with open(metadata_file, 'r') as metadata_f:
                            metadata_content = metadata_f.read()
                            metadata_dict = json.loads(metadata_content)
                    except Exception as e:
                        _LOGGER.warning(f"Failed to read metadata file {metadata_file}: {e}")

                # Extract modality from metadata if available
                if metadata_dict is not None:
                    metadata_dict_lower = {k.lower(): v for k, v in metadata_dict.items() if isinstance(k, str)}
                    try:
                        if modality is None:
                            if 'modality' in metadata_dict_lower:
                                modality = metadata_dict_lower['modality']
                    except Exception as e:
                        _LOGGER.debug(f"Failed to extract modality from metadata: {e}")

            form = aiohttp.FormData()
            url = self._get_endpoint_url(RootAPIHandler.ENDPOINT_RESOURCES)
            file_key = 'resource'
            form.add_field('source', 'api')

            form.add_field(file_key, f, filename=filename, content_type=mimetype)
            form.add_field('source_filepath', name)  # full path to the file
            if mimetype is not None:
                form.add_field('mimetype', mimetype)
            if channel is not None:
                form.add_field('channel', channel)
            if modality is not None:
                form.add_field('modality', modality)
            form.add_field('bypass_inbox', 'true' if publish else 'false')
            if tags is not None and len(tags) > 0:
                # comma separated list of tags
                form.add_field('tags', ','.join([l.strip() for l in tags]))

            # Add JSON metadata if provided
            if metadata_content is not None:
                try:
                    _LOGGER.debug("Adding metadata to form data")
                    form.add_field('metadata', metadata_content, content_type='application/json')
                except Exception as e:
                    _LOGGER.warning(f"Failed to add metadata to form: {e}")

            request_params = {
                'method': 'POST',
                'url': url,
                'data': form
            }

            resp_data = await self._run_request_async(request_params, session)
            if 'error' in resp_data:
                raise DatamintException(resp_data['error'])
            _LOGGER.info(f"Response on uploading {name}: {resp_data}")

            _USER_LOGGER.info(f'"{name}" uploaded')
            return resp_data['id']
        except Exception as e:
            if 'name' in locals():
                _LOGGER.error(f"Error uploading {name}: {e}")
            else:
                _LOGGER.error(f"Error uploading {file_path}: {e}")
            raise e
        finally:
            f.close()

    async def _upload_resources_async(self,
                                      files_path: Sequence[str | IO],
                                      mimetype: Optional[str] = None,
                                      anonymize: bool = False,
                                      anonymize_retain_codes: Sequence[tuple] = [],
                                      on_error: Literal['raise', 'skip'] = 'raise',
                                      tags=None,
                                      mung_filename: Sequence[int] | Literal['all'] = None,
                                      channel: Optional[str] = None,
                                      modality: Optional[str] = None,
                                      publish: bool = False,
                                      segmentation_files: Optional[list[dict]] = None,
                                      transpose_segmentation: bool = False,
                                      metadata_files: Optional[list[str | dict | None]] = None,
                                      ) -> list[str]:
        if on_error not in ['raise', 'skip']:
            raise ValueError("on_error must be either 'raise' or 'skip'")

        if segmentation_files is None:
            segmentation_files = _infinite_gen(None)

        if metadata_files is None:
            metadata_files = _infinite_gen(None)

        async with aiohttp.ClientSession() as session:
            async def __upload_single_resource(file_path, segfiles: dict[str, list | dict], metadata_file: str | dict | None):
                rid = await self._upload_single_resource_async(
                    file_path=file_path,
                    mimetype=mimetype,
                    anonymize=anonymize,
                    anonymize_retain_codes=anonymize_retain_codes,
                    tags=tags,
                    session=session,
                    mung_filename=mung_filename,
                    channel=channel,
                    modality=modality,
                    publish=publish,
                    metadata_file=metadata_file,
                )
                if segfiles is not None:
                    fpaths = segfiles['files']
                    names = segfiles.get('names', _infinite_gen(None))
                    if isinstance(names, dict):
                        names = _infinite_gen(names)
                    frame_indices = segfiles.get('frame_index', _infinite_gen(None))
                    for f, name, frame_index in tqdm(zip(fpaths, names, frame_indices),
                                                     desc=f"Uploading segmentations for {file_path}",
                                                     total=len(fpaths)):
                        if f is not None:
                            await self._upload_segmentations_async(rid,
                                                                   file_path=f,
                                                                   name=name,
                                                                   frame_index=frame_index,
                                                                   transpose_segmentation=transpose_segmentation)
                return rid

            tasks = [__upload_single_resource(f, segfiles, metadata_file)
                     for f, segfiles, metadata_file in zip(files_path, segmentation_files, metadata_files)]
            return await asyncio.gather(*tasks, return_exceptions=on_error == 'skip')

    def _assemble_dicoms(self, files_path: Sequence[str | IO]) -> tuple[Sequence[str | IO], bool]:
        dicoms_files_path = []
        other_files_path = []
        for f in files_path:
            if is_dicom(f):
                dicoms_files_path.append(f)
            else:
                other_files_path.append(f)

        orig_len = len(dicoms_files_path)
        if orig_len == 0:
            _LOGGER.debug("No DICOM files found to assemble.")
            return files_path, False
        dicoms_files_path = dicom_utils.assemble_dicoms(dicoms_files_path, return_as_IO=True)

        new_len = len(dicoms_files_path)
        if new_len != orig_len:
            _LOGGER.info(f"Assembled {new_len} dicom files out of {orig_len} files.")
            files_path = itertools.chain(dicoms_files_path, other_files_path)
            assembled = True
        else:
            assembled = False

        return files_path, assembled

    def upload_resource(self,
                        file_path: str | IO | pydicom.dataset.Dataset,
                        mimetype: Optional[str] = None,
                        anonymize: bool = False,
                        anonymize_retain_codes: Sequence[tuple] = [],
                        on_error: Literal['raise', 'skip'] = 'raise',
                        tags: Optional[Sequence[str]] = None,
                        mung_filename: Sequence[int] | Literal['all'] | None = None,
                        channel: Optional[str] = None,
                        publish: bool = False,
                        publish_to: Optional[str] = None,
                        segmentation_files: list[str] | dict | None = None,
                        transpose_segmentation: bool = False,
                        modality: Optional[str] = None,
                        assemble_dicoms: bool = True,
                        metadata: dict | str | None = None
                        ) -> str | Exception:
        """
        Uploads a single resource (examples: DICOM file, NIfTI file, pydicom object)

        Args:
            file_path: The path to the resource file or a list of paths to resources files.
            mimetype: The mimetype of the resources. If None, it will be guessed.
            anonymize: Whether to anonymize the dicoms or not.
            anonymize_retain_codes: The tags to retain when anonymizing the dicoms.
            on_error: Whether to raise an exception when an error occurs or to skip the error.
            tags: The tags to add to the resources.
            mung_filename: The parts of the filepath to keep when renaming the resource file.
                ''all'' keeps all parts.
            channel: The channel to upload the resources to. An arbitrary name to group the resources.
            publish: Whether to directly publish the resources or not. They will have the 'published' status.
            publish_to: The project name or id to publish the resources to.
                They will have the 'published' status and will be added to the project.
                If this is set, `publish` parameter is ignored.
            segmentation_files: The segmentation files to upload. 
                If it is a dict, it should have two keys: 'files' and 'names'.
                    - files: A list of paths to the segmentation files. Example: ['seg1.nii.gz', 'seg2.nii.gz'].
                    - names: Can be a list (same size of `files`) of labels for the segmentation files. Example: ['Brain', 'Lung']. Or a dictionary mapping a pixel value to a label. Example: {'1': 'Brain', '2': 'Lung'}.
            transpose_segmentation: Whether to transpose the segmentation files or not.
            modality: The modality of the resources.
            assemble_dicoms: Whether to assemble the dicom files or not based on the SeriesInstanceUID and InstanceNumber attributes.
            metadata: JSON metadata to include with each resource.
                Can be a file path (str) or an already loaded dictionary (dict).

        Raises:
            ResourceNotFoundError: If `publish_to` is supplied, and the project does not exists.

        Returns:
            str | Exception: A resource ID or an error.

        Example:
            .. code-block:: python

                file_path = '/path/to/resource.dcm'
                segfiles = {'files': ['/path/to/seg_brain.nii.gz', '/path/to/seg_lung.nii.gz'],
                            'names': ['Brain', 'Lung']}
                result = api_handler.upload_resource(
                    file_path=file_path,
                    mimetype='application/dicom',
                    anonymize=True,
                    tags=['example', 'dicom'],
                    segmentation_files=segfiles,
                    metadata={'PatientID': '12345'}
                )
        """

        result = self.upload_resources(
            files_path=[file_path],
            mimetype=mimetype,
            anonymize=anonymize,
            anonymize_retain_codes=anonymize_retain_codes,
            on_error=on_error,
            tags=tags,
            mung_filename=mung_filename,
            channel=channel,
            publish=publish,
            publish_to=publish_to,
            segmentation_files=[segmentation_files] if segmentation_files is not None else None,
            transpose_segmentation=transpose_segmentation,
            modality=modality,
            assemble_dicoms=assemble_dicoms,
            metadata=metadata
        )

        return result[0]

    @staticmethod
    def _is_dicom_report(file_path: str | IO) -> bool:
        """
        Check if a DICOM file is a report (e.g., Structured Report).

        Args:
            file_path: Path to the DICOM file or file-like object.

        Returns:
            bool: True if the DICOM file is a report, False otherwise.
        """
        try:
            if not is_dicom(file_path):
                return False

            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            if hasattr(file_path, 'seek'):
                file_path.seek(0)
            modality = getattr(ds, 'Modality', None)

            # Common report modalities
            # SR=Structured Report, DOC=Document, KO=Key Object, PR=Presentation State
            report_modalities = {'SR', 'DOC', 'KO', 'PR', 'ESR'}

            return modality in report_modalities
        except Exception as e:
            _LOGGER.warning(f"Error checking if DICOM is a report: {e}")
            return False

    def upload_resources(self,
                         files_path: str | IO | Sequence[str | IO] | pydicom.dataset.Dataset,
                         mimetype: Optional[str] = None,
                         anonymize: bool = False,
                         anonymize_retain_codes: Sequence[tuple] = [],
                         on_error: Literal['raise', 'skip'] = 'raise',
                         tags: Optional[Sequence[str]] = None,
                         mung_filename: Sequence[int] | Literal['all'] | None = None,
                         channel: Optional[str] = None,
                         publish: bool = False,
                         publish_to: Optional[str] = None,
                         segmentation_files: Optional[list[Union[list[str], dict]]] = None,
                         transpose_segmentation: bool = False,
                         modality: Optional[str] = None,
                         assemble_dicoms: bool = True,
                         metadata: list[str | dict | None] | dict | str | None = None,
                         discard_dicom_reports: bool = True
                         ) -> list[str | Exception] | str | Exception:
        """
        Upload resources.

        Args:
            files_path (str | IO | Sequence[str | IO]): The path to the resource file or a list of paths to resources files.
            mimetype (str): The mimetype of the resources. If None, it will be guessed.
            anonymize (bool): Whether to anonymize the dicoms or not.
            anonymize_retain_codes (Sequence[tuple]): The tags to retain when anonymizing the dicoms.
            on_error (Literal['raise', 'skip']): Whether to raise an exception when an error occurs or to skip the error.
            tags (Optional[Sequence[str]]): The tags to add to the resources.
            mung_filename (Sequence[int] | Literal['all']): The parts of the filepath to keep when renaming the resource file.
                ''all'' keeps all parts.
            channel (Optional[str]): The channel to upload the resources to. An arbitrary name to group the resources.
            publish (bool): Whether to directly publish the resources or not. They will have the 'published' status.
            publish_to (Optional[str]): The project name or id to publish the resources to.
                They will have the 'published' status and will be added to the project.
                If this is set, `publish` parameter is ignored.
            segmentation_files (Optional[list[Union[list[str], dict]]]): The segmentation files to upload.
                If each element is a dict, it should have two keys: 'files' and 'names'.
                    - files: A list of paths to the segmentation files. Example: ['seg1.nii.gz', 'seg2.nii.gz'].
                    - names: Can be a list (same size of `files`) of labels for the segmentation files. Example: ['Brain', 'Lung']. 
            transpose_segmentation (bool): Whether to transpose the segmentation files or not.
            modality (Optional[str]): The modality of the resources.
            assemble_dicoms (bool): Whether to assemble the dicom files or not based on the SeriesInstanceUID and InstanceNumber attributes.
            metadatas (Optional[list[str | dict | None]]): JSON metadata to include with each resource.
                Must have the same length as `files_path`.
                Can be file paths (str) or already loaded dictionaries (dict).

        Raises:
            ResourceNotFoundError: If `publish_to` is supplied, and the project does not exists.

        Returns:
            list[str | Exception]: A list of resource IDs or errors.
        """

        if on_error not in ['raise', 'skip']:
            raise ValueError("on_error must be either 'raise' or 'skip'")

        files_path, is_multiple_resources = RootAPIHandler.__process_files_parameter(files_path)

        ### Discard DICOM reports
        if discard_dicom_reports:
            files_path = [f for f in files_path if not RootAPIHandler._is_dicom_report(f)]
            old_size = len(files_path)
            if old_size is not None and old_size != len(files_path):
                _LOGGER.info(f"Discarded {old_size - len(files_path)} DICOM report files from upload.")

        if isinstance(metadata, (str, dict)):
            _LOGGER.debug("Converting metadatas to a list")
            metadata = [metadata]

        if metadata is not None and len(metadata) != len(files_path):
            raise ValueError("The number of metadata files must match the number of resources.")
        if assemble_dicoms:
            files_path, assembled = self._assemble_dicoms(files_path)
            assemble_dicoms = assembled

        if segmentation_files is not None:
            if assemble_dicoms:
                raise NotImplementedError("Segmentation files cannot be uploaded when assembling dicoms yet.")
            if is_multiple_resources:
                if len(segmentation_files) != len(files_path):
                    raise ValueError("The number of segmentation files must match the number of resources.")
            else:
                if isinstance(segmentation_files, list) and isinstance(segmentation_files[0], list):
                    raise ValueError("segmentation_files should not be a list of lists if files_path is not a list.")
                if isinstance(segmentation_files, dict):
                    segmentation_files = [segmentation_files]

            segmentation_files = [segfiles if (isinstance(segfiles, dict) or segfiles is None) else {'files': segfiles}
                                  for segfiles in segmentation_files]

            for segfiles in segmentation_files:
                if segfiles is None:
                    continue
                if 'files' not in segfiles:
                    raise ValueError("segmentation_files must contain a 'files' key with a list of file paths.")
                if 'names' in segfiles:
                    # same length as files
                    if isinstance(segfiles['names'], (list, tuple)) and len(segfiles['names']) != len(segfiles['files']):
                        raise ValueError(
                            "segmentation_files['names'] must have the same length as segmentation_files['files'].")

        loop = asyncio.get_event_loop()
        task = self._upload_resources_async(files_path=files_path,
                                            mimetype=mimetype,
                                            anonymize=anonymize,
                                            anonymize_retain_codes=anonymize_retain_codes,
                                            on_error=on_error,
                                            tags=tags,
                                            mung_filename=mung_filename,
                                            channel=channel,
                                            publish=publish,
                                            segmentation_files=segmentation_files,
                                            transpose_segmentation=transpose_segmentation,
                                            modality=modality,
                                            metadata_files=metadata,
                                            )

        resource_ids = loop.run_until_complete(task)
        _LOGGER.info(f"Resources uploaded: {resource_ids}")

        if publish_to is not None:
            _USER_LOGGER.info('Adding resources to project')
            resource_ids_succ = [rid for rid in resource_ids if not isinstance(rid, Exception)]
            try:
                self.add_to_project(resource_ids_succ, publish_to)
            except Exception as e:
                _LOGGER.error(f"Error adding resources to project: {e}")
                if on_error == 'raise':
                    raise e

        if is_multiple_resources:
            return resource_ids
        return resource_ids[0]

    def publish_resources(self,
                          resource_ids: str | Sequence[str],
                          project_name: Optional[str] = None,
                          ) -> None:
        """
        Publish a resource, changing its status to 'published'.

        Args:
            resource_ids (str|Sequence[str]): The resource unique id or a list of resource unique ids.
            project_name (str): The project name or id to publish the resource to.

        Raises:
            ResourceNotFoundError: If the resource does not exists or the project does not exists.

        """
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]

        for resource_id in resource_ids:
            params = {
                'method': 'POST',
                'url': f'{self.root_url}/resources/{resource_id}/publish',
            }

            try:
                self._run_request(params)
            except ResourceNotFoundError as e:
                e.set_params('resource', {'resource_id': resource_id})
                raise e
            except HTTPError as e:
                if project_name is None and BaseAPIHandler._has_status_code(e, 400) and 'Resource must be in inbox status to be approved' in e.response.text:
                    _LOGGER.warning(f"Resource {resource_id} is not in inbox status. Skipping publishing")
                else:
                    raise e

        if project_name is None:
            return

        # get the project id by its name
        project = self.get_project_by_name(project_name)
        if 'error' in project:
            raise ResourceNotFoundError('project', {'project_name': project_name})

        dataset_id = project['dataset_id']

        params = {
            'method': 'POST',
            'url': f'{self.root_url}/datasets/{dataset_id}/resources',
            'json': {'resource_ids_to_add': resource_ids, 'all_files_selected': False}
        }

        self._run_request(params)

    def add_to_project(self,
                       resource_ids: str | Sequence[str],
                       project_name: str,
                       ) -> None:
        """
        Add resources to a project.

        Args:
            resource_ids (str|Sequence[str]): The resource unique id or a list of resource unique ids.
            project_name (str): The project name or id to add the resource to.

        Raises:
            ResourceNotFoundError: If the resource does not exists or the project does not exists.

        """
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]

        # get the project id by its name
        project = self.get_project_by_name(project_name)
        if 'error' in project:
            raise ResourceNotFoundError('project', {'project_name': project_name})

        dataset_id = project['dataset_id']

        params = {
            'method': 'POST',
            'url': f'{self.root_url}/datasets/{dataset_id}/resources',
            'json': {'resource_ids_to_add': resource_ids, 'all_files_selected': False}
        }

        self._run_request(params)

    def get_project_by_id(self, project_id: str) -> dict:
        """
        Get a project by its id.

        Args:
            project_id (str): The project id.

        Returns:
            dict: The project information.

        Raises:
            ResourceNotFoundError: If the project does not exists.
        """
        try:
            request_params = {
                'method': 'GET',
                'url': f'{self.root_url}/projects/{project_id}',
            }
            return self._run_request(request_params).json()
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 500:
                raise ResourceNotFoundError('project', {'project_id': project_id})
            raise e

    def get_project_by_name(self, project_name: str) -> dict:
        """
        Get a project by its name.

        Args:
            project_name (str): The project name.

        Returns:
            dict: The project information.

        Raises:
            ResourceNotFoundError: If the project does not exists.
        """
        try:
            all_projects = self.get_projects()
            for project in all_projects:
                if project['name'] == project_name or project['id'] == project_name:
                    return project
            return {'error': 'No project with specified name found',
                    'all_projects': [project['name'] for project in all_projects]}

        except ResourceNotFoundError as e:
            e.set_params('project', {'project_name': project_name})
            raise e

    @staticmethod
    def __process_files_parameter(file_path: str | IO | Sequence[str | IO] | pydicom.dataset.Dataset) -> tuple[Sequence[str | IO], bool]:
        """
        Process the file_path parameter to ensure it is a list of file paths or IO objects.
        """
        if isinstance(file_path, pydicom.dataset.Dataset):
            file_path = to_bytesio(file_path, file_path.filename)

        if isinstance(file_path, str):
            if os.path.isdir(file_path):
                is_list = True
                new_file_path = [f'{file_path}/{f}' for f in os.listdir(file_path)]
            else:
                is_list = False
                new_file_path = [file_path]
        # Check if is an IO object
        elif _is_io_object(file_path):
            is_list = False
            new_file_path = [file_path]
        elif not hasattr(file_path, '__len__'):
            if hasattr(file_path, '__iter__'):
                is_list = True
                new_file_path = list(file_path)
            else:
                is_list = False
                new_file_path = [file_path]
        else:
            is_list = True
            new_file_path = file_path
        return new_file_path, is_list

    def get_resources_by_ids(self, ids: str | Sequence[str]) -> dict[str, Any] | Sequence[dict[str, Any]]:
        """
        Get resources by their unique ids.

        Args:
            ids (str | Sequence[str]): The resource unique id or a list of resource unique ids.

        Returns:
            dict | Sequence[dict]: The resource information or a list of resource information.

        Raises:
            ResourceNotFoundError: If the resource does not exists.

        """
        input_is_a_string = isinstance(ids, str)  # used later to return a single object or a list of objects
        if input_is_a_string:
            ids = [ids]

        resources = []
        try:
            for i in ids:
                request_params = {
                    'method': 'GET',
                    'url': f'{self.root_url}/resources/{i}',
                }

                resources.append(self._run_request(request_params).json())
        except ResourceNotFoundError as e:
            e.set_params('resource', {'resource_id': i})
            raise e

        return resources[0] if input_is_a_string else resources

    def get_resources(self,
                      status: Optional[ResourceStatus] = None,
                      from_date: Optional[date] = None,
                      to_date: Optional[date] = None,
                      tags: Optional[Sequence[str]] = None,
                      modality: Optional[str] = None,
                      mimetype: Optional[str] = None,
                      return_ids_only: bool = False,
                      order_field: Optional[ResourceFields] = None,
                      order_ascending: Optional[bool] = None,
                      channel: Optional[str] = None,
                      project_name: str | list[str] | None = None,
                      filename: Optional[str] = None
                      ) -> Generator[dict, None, None]:
        """
        Iterates over resources with the specified filters.
        Filters can be combined to narrow down the search.
        It returns full information of the resources by default, but it can be configured to return only the ids with parameter `return_ids_only`.

        Args:
            status (ResourceStatus): The resource status. Possible values: 'inbox', 'published', 'archived' or None. If None, it will return all resources.
            from_date (Optional[date]): The start date.
            to_date (Optional[date]): The end date.
            tags (Optional[list[str]]): The tags to filter the resources.
            modality (Optional[str]): The modality of the resources.
            mimetype (Optional[str]): The mimetype of the resources.
            return_ids_only (bool): Whether to return only the ids of the resources.
            order_field (Optional[ResourceFields]): The field to order the resources. See :data:`~.base_api_handler.ResourceFields`.
            order_ascending (Optional[bool]): Whether to order the resources in ascending order.
            project_name (str | list[str] | None): The project name or a list of project names to filter resources by project.
                If multiple projects are provided, resources will be filtered to include only those belonging to ALL of the specified projects.

        Returns:
            Generator[dict, None, None]: A generator of dictionaries with the resources information.

        Example:
            >>> for resource in api_handler.get_resources(status='inbox'):
            >>>     print(resource)
        """
        # Convert datetime objects to ISO format
        if from_date:
            from_date = from_date.isoformat()
        if to_date:
            to_date = to_date.isoformat()

        # Prepare the payload
        payload = {
            "from": from_date,
            "to": to_date,
            "status": status if status is not None else "",
            "modality": modality,
            "mimetype": mimetype,
            "ids": return_ids_only,
            "order_field": order_field,
            "order_by_asc": order_ascending,
            "channel_name": channel,
            "filename": filename,
        }
        if project_name is not None:
            if isinstance(project_name, str):
                project_name = [project_name]
            payload["project"] = json.dumps({'items': project_name,
                                             'filterType': 'intersection'})  # union or intersection

        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            tags_filter = {
                "items": tags,
                "filterType": "union"
            }
            payload['tags'] = json.dumps(tags_filter)

        # Remove None values from the payload.
        # Maybe it is not necessary.
        for k in list(payload.keys()):
            if payload[k] is None:
                del payload[k]

        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/resources',
            'params': payload
        }

        yield from self._run_pagination_request(request_params,
                                                return_field=['data', 0, 'resources'])

    def get_channels(self) -> Generator[dict, None, None]:
        """
        Iterates over the channels with the specified filters.

        Returns:
           Generator[dict, None, None]: A generator of dictionaries with the channels information.

        Example:
            >>> list(api_handler.get_channels()) # Gets all channels
            [{'channel_name': 'test_channel',
                'resource_data': [{'created_by': 'datamint-dev@mail.com',
                                    'customer_id': '79113ed1-0535-4f53-9359-7fe3fa9f28a8',
                                    'resource_id': 'a05fe46d-2f66-46fc-b7ef-666464ad3a28',
                                    'resource_file_name': 'image.png',
                                    'resource_mimetype': 'image/png'}],
                'deleted': False,
                'created_at': '2024-06-04T12:38:12.976Z',
                'updated_at': '2024-06-04T12:38:12.976Z',
                'resource_count': '1'}]

        """

        request_params = {
            'method': 'GET',
            'url': self._get_endpoint_url(RootAPIHandler.ENDPOINT_CHANNELS),
            'params': {}
        }

        yield from self._run_pagination_request(request_params,
                                                return_field='data')

    def set_resource_tags(self,
                          resource_id: str,
                          tags: Sequence[str],
                          ):
        url = f"{self._get_endpoint_url(RootAPIHandler.ENDPOINT_RESOURCES)}/{resource_id}/tags"
        data = {}

        if tags is not None:
            data['tags'] = tags

        request_params = {'method': 'PUT',
                          'url': url,
                          'json': data
                          }

        response = self._run_request(request_params)
        return response

    @staticmethod
    def _has_status_code(e, status_code: int) -> bool:
        return hasattr(e, 'response') and (e.response is not None) and e.response.status_code == status_code

    async def _async_download_file(self,
                                   resource_id: str,
                                   save_path: str,
                                   session: aiohttp.ClientSession | None = None,
                                   progress_bar: tqdm | None = None):
        """
        Asynchronously download a file from the server.

        Args:
            resource_id (str): The resource unique id.
            save_path (str): The path to save the file.
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            progress_bar (tqdm | None): Optional progress bar to update after download completion.
        """
        url = f"{self._get_endpoint_url(RootAPIHandler.ENDPOINT_RESOURCES)}/{resource_id}/file"
        request_params = {
            'method': 'GET',
            'headers': {'accept': 'application/octet-stream'},
            'url': url
        }
        try:
            data_bytes = await self._run_request_async(request_params, session, 'content')
            with open(save_path, 'wb') as f:
                f.write(data_bytes)
            if progress_bar:
                progress_bar.update(1)
        except ResourceNotFoundError as e:
            e.set_params('resource', {'resource_id': resource_id})
            raise e

    def download_multiple_resources(self,
                                    resource_ids: list[str],
                                    save_path: list[str] | str
                                    ) -> None:
        """
        Download multiple resources and save them to the specified paths.

        Args:
            resource_ids (list[str]): A list of resource unique ids.
            save_path (list[str] | str): A list of paths to save the files or a directory path.
        """
        async def _download_all_async():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._async_download_file(resource_id, save_path=path, session=session, progress_bar=progress_bar)
                    for resource_id, path in zip(resource_ids, save_path)
                ]
                await asyncio.gather(*tasks)

        if isinstance(save_path, str):
            save_path = [os.path.join(save_path, r) for r in resource_ids]

        with tqdm(total=len(resource_ids), desc="Downloading resources", unit="file") as progress_bar:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_download_all_async())

    def download_resource_file(self,
                               resource_id: str,
                               save_path: Optional[str] = None,
                               auto_convert: bool = True,
                               add_extension: bool = False
                               ) -> bytes | pydicom.dataset.Dataset | Image.Image | cv2.VideoCapture | nib_FileBasedImage | tuple[Any, str]:
        """
        Download a resource file.

        Args:
            resource_id (str): The resource unique id.
            save_path (Optional[str]): The path to save the file.
            auto_convert (bool): Whether to convert the file to a known format or not.
            add_extension (bool): Whether to add the appropriate file extension to the save_path based on the content type.

        Returns:
            The resource content in bytes (if `auto_convert=False`) or the resource object (if `auto_convert=True`).
            if `add_extension=True`, the function will return a tuple of (resource_data, save_path).

        Raises:
            ResourceNotFoundError: If the resource does not exists.

        Example:
            >>> api_handler.download_resource_file('resource_id', auto_convert=False)
                returns the resource content in bytes.
            >>> api_handler.download_resource_file('resource_id', auto_convert=True)
                Assuming this resource is a dicom file, it will return a pydicom.dataset.Dataset object. 
            >>> api_handler.download_resource_file('resource_id', save_path='path/to/dicomfile.dcm')
                saves the file in the specified path.
        """
        url = f"{self._get_endpoint_url(RootAPIHandler.ENDPOINT_RESOURCES)}/{resource_id}/file"
        request_params = {'method': 'GET',
                          'headers': {'accept': 'application/octet-stream'},
                          'url': url}
        try:
            response = self._run_request(request_params)

            # Get mimetype if needed for auto_convert or add_extension
            mimetype = None
            if auto_convert or add_extension:
                resource_info = self.get_resources_by_ids(resource_id)
                mimetype = resource_info['mimetype']

            if auto_convert:
                try:
                    resource_file = BaseAPIHandler.convert_format(response.content,
                                                                  mimetype,
                                                                  save_path)
                except ValueError as e:
                    _LOGGER.warning(f"Could not convert file to a known format: {e}")
                    resource_file = response.content
                except NotImplementedError as e:
                    _LOGGER.warning(f"Conversion not implemented yet for {mimetype} and save_path=None." +
                                    " Returning a bytes array. If you want the conversion for this mimetype, provide a save_path.")
                    resource_file = response.content
            else:
                resource_file = response.content
        except ResourceNotFoundError as e:
            e.set_params('resource', {'resource_id': resource_id})
            raise e

        if save_path is not None:
            if add_extension:
                ext = mimetypes.guess_extension(mimetype)
                if ext is not None and not save_path.endswith(ext):
                    save_path += ext
            with open(save_path, 'wb') as f:
                f.write(response.content)

        if add_extension:
            return resource_file, save_path
        return resource_file

    def download_resource_frame(self,
                                resource_id: str,
                                frame_index: int) -> Image.Image:
        """
        Download a frame of a resource.
        This is faster than downloading the whole resource and then extracting the frame.

        Args:
            resource_id: The resource unique id.
            frame_index: The index of the frame to download.

        Returns:
            Image.Image: The frame as a PIL image.

        Raises:
            ResourceNotFoundError: If the resource does not exists.
            DatamintException: If the resource is not a video or dicom.
        """
        # check if the resource is an single frame image (png,jpeg,...) first.
        # If so, download the whole resource file and return the image.
        resource_info = self.get_resources_by_ids(resource_id)
        if resource_info['mimetype'].startswith('image/') or resource_info.get('storage') == 'ImageResource':
            if frame_index != 0:
                raise DatamintException(f"Resource {resource_id} is a single frame image, "
                                        f"but frame_index is {frame_index}.")
            return self.download_resource_file(resource_id, auto_convert=True)

        url = f"{self._get_endpoint_url(RootAPIHandler.ENDPOINT_RESOURCES)}/{resource_id}/frames/{frame_index}"
        request_params = {'method': 'GET',
                          'headers': {'accept': 'image/png'},
                          'url': url}
        try:
            response = self._run_request(request_params)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            else:
                raise DatamintException(
                    f"Error downloading frame {frame_index} of resource {resource_id}: {response.text}")
        except ResourceNotFoundError as e:
            e.set_params('resource', {'resource_id': resource_id})
            raise e

    def delete_resources(self, resource_ids: Sequence[str] | str) -> None:
        """
        Delete resources by their unique ids.

        Args:
            resource_ids (Sequence[str] | str): The resource unique id or a list of resource unique ids.

        Raises:
            ResourceNotFoundError: If the resource does not exists.

        Example:
            >>> api_handler.delete_resources('e8b78358-656d-481f-8c98-d13b9ba6be1b')
            >>> api_handler.delete_resources(['e8b78358-656d-481f-8c98-d13b9ba6be1b', '6f8b506c-6ea1-4e85-8e67-254767f95a7b'])
        """
        if isinstance(resource_ids, str):
            resource_ids = [resource_ids]
        for rid in resource_ids:
            url = f"{self._get_endpoint_url(RootAPIHandler.ENDPOINT_RESOURCES)}/{rid}"
            request_params = {'method': 'DELETE',
                              'url': url
                              }
            try:
                self._run_request(request_params)
            except ResourceNotFoundError as e:
                e.set_params('resource', {'resource_id': rid})
                raise e

    def get_datasets(self) -> list[dict]:
        """
        Get all datasets.

        Returns:
            list[dict]: A list of dictionaries with the datasets information.

        """
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/datasets',
        }

        response = self._run_request(request_params)
        return response.json()['data']

    @deprecated(version='1.7')
    def get_datasetsinfo_by_name(self, dataset_name: str) -> list[dict]:
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/datasets',
        }
        # FIXME: inefficient to get all datasets and then filter by name
        resp = self._run_request(request_params).json()
        datasets = [d for d in resp['data'] if d['name'] == dataset_name]
        return datasets

    def get_dataset_by_id(self, dataset_id: str) -> dict:
        try:
            request_params = {
                'method': 'GET',
                'url': f'{self.root_url}/datasets/{dataset_id}',
            }
            return self._run_request(request_params).json()
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 500:
                raise ResourceNotFoundError('dataset', {'dataset_id': dataset_id})
            raise e

    def get_users(self) -> list[dict]:
        """
        Get all users.

        Returns:
            list[dict]: A list of dictionaries with the users information.

        """
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/users',
        }

        response = self._run_request(request_params)
        return response.json()

    def create_user(self,
                    email: str,
                    password: Optional[str] = None,
                    firstname: Optional[str] = None,
                    lastname: Optional[str] = None,
                    roles: Optional[list[str]] = None) -> dict:
        """
        Create a user.

        Args:
            email (str): The user email.
            password (Optional[str]): The user password.
            firstname (Optional[str]): The user first name.
            lastname (Optional[str]): The user last name.
            roles (Optional[list[str]]): The user roles. If None, the user will be created with the default role.

        Returns:
            dict: The user information.
        """

        request_params = {
            'method': 'POST',
            'url': f'{self.root_url}/users',
            'json': {'email': email, 'password': password, 'firstname': firstname, 'lastname': lastname, 'roles': roles}
        }

        try:
            resp = self._run_request(request_params)
            return resp.json()
        except HTTPError as e:
            _LOGGER.error(f"Error creating user: {e.response.text}")
            raise e

    def get_projects(self) -> list[dict]:
        """
        Get the list of projects.

        Returns:
            list[dict]: The list of projects.

        Example:
            >>> api_handler.get_projects()
            [{'id': '15ab9105-6e92-48c0-bb21-8e1325ec4305',
            'name': 'Example Project',
            'description': 'this is an example project',
            'created_at': '2025-04-23T14:41:03.475Z',
            'created_by': 'user@mail.com',
            (...)}
            ]
        """
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/projects'
        }
        return self._run_request(request_params).json()['data']

    def get_project_resources(self, project_id: str) -> list[dict]:
        """
        Get the resources of a project by its id.

        Args:
            project_id (str): The project id.

        Returns:
            list[dict]: The list of resources in the project.

        Raises:
            ResourceNotFoundError: If the project does not exists.
        """
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/projects/{project_id}/resources'
        }
        try:
            return self._run_request(request_params).json()
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 500:
                raise ResourceNotFoundError('project', {'project_id': project_id})
            raise e

    def create_project(self,
                       name: str,
                       description: str,
                       resources_ids: list[str],
                       is_active_learning: bool = False,
                       two_up_display: bool = False,
                       ) -> dict:
        """
        Create a new project.

        Args:
            name (str): The name of the project.
            description (str): The description of the project.
            resources_ids (list[str]): The list of resource ids to be included in the project.
            is_active_learning (bool): Whether the project is an active learning project or not.
            two_up_display (bool): allow annotators to display multiple resources for annotation.

        Returns:
            dict: The created project.

        Raises:
            DatamintException: If the project could not be created.
        """

        request_args = {
            'url': self._get_endpoint_url('projects'),
            'method': 'POST',
            'json': {'name': name,
                     'is_active_learning': is_active_learning,
                     'resource_ids': resources_ids,
                     'annotation_set': {
                         "annotators": [],
                         "resource_ids": resources_ids,
                         "annotations": [],
                         "frame_labels": [],
                         "image_labels": [],
                     },
                     "two_up_display": two_up_display,
                     "require_review": False,
                     'description': description}
        }
        response = self._run_request(request_args)
        self._check_errors_response_json(response)
        return response.json()

    def delete_project(self, project_id: str) -> None:
        """
        Delete a project by its id.

        Args:
            project_id (str): The project id.

        Raises:
            ResourceNotFoundError: If the project does not exists.
        """
        url = f"{self._get_endpoint_url('projects')}/{project_id}"
        request_params = {'method': 'DELETE',
                          'url': url
                          }
        try:
            resp = self._run_request(request_params)
            _LOGGER.debug(resp)
        except ResourceNotFoundError as e:
            e.set_params('project', {'project_id': project_id})
            raise e

    def download_project(self, project_id: str,
                         outpath: str,
                         all_annotations: bool = False,
                         include_unannotated: bool = False,
                         ) -> None:
        """
        Download a project by its id.

        Args:
            project_id (str): The project id.
            outpath (str): The path to save the project zip file.
            all_annotations (bool): Whether to include all annotations in the downloaded dataset, 
                even those not made by the provided project.
            include_unannotated (bool): Whether to include unannotated resources in the downloaded dataset.

        Example:
            >>> api_handler.download_project('project_id', 'path/to/project.zip')
        """
        url = f"{self._get_endpoint_url('projects')}/{project_id}/annotated_dataset"
        request_params = {'method': 'GET',
                          'url': url,
                          'stream': True,
                          'params': {'all_annotations': all_annotations}
                          }
        if include_unannotated:
            request_params['params']['include_unannotated'] = include_unannotated

        response = self._run_request(request_params)
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0:
            total_size = None
        with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
            with open(outpath, 'wb') as file:
                for data in response.iter_content(1024):
                    progress_bar.update(len(data))
                    file.write(data)
