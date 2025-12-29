from typing import TypeAlias, Literal, IO
from collections.abc import Sequence
from ..base_api import ApiConfig, BaseApi
from ..entity_base_api import CreatableEntityApi, DeletableEntityApi
from datamint.entities.resource import Resource
from datamint.entities.annotations.annotation import Annotation
from datamint.exceptions import DatamintException, ResourceNotFoundError
from datamint.api.dto import AnnotationType
import httpx
from datetime import date
import json
import logging
import pydicom
from medimgkit.dicom_utils import anonymize_dicom, to_bytesio, is_dicom, is_dicom_report, GeneratorWithLength
from medimgkit import dicom_utils, standardize_mimetype
from medimgkit.io_utils import is_io_object, peek
from medimgkit.format_detection import guess_typez, guess_extension, DEFAULT_MIME_TYPE
from medimgkit.nifti_utils import DEFAULT_NIFTI_MIME, NIFTI_MIMES
import os
import itertools
from tqdm.auto import tqdm
import asyncio
import aiohttp
from pathlib import Path
import nest_asyncio  # For running asyncio in jupyter notebooks
from PIL import Image
import io
from datamint.types import ImagingData
from collections import defaultdict


_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')

ResourceStatus: TypeAlias = Literal['new', 'inbox', 'published', 'archived']
"""TypeAlias: The available resource status. Possible values: 'new', 'inbox', 'published', 'archived'.
"""
ResourceFields: TypeAlias = Literal['modality', 'created_by', 'published_by', 'published_on', 'filename', 'created_at']
"""TypeAlias: The available fields to order resources. Possible values: 'modality', 'created_by', 'published_by', 'published_on', 'filename', 'created_at' (default).
"""


def _infinite_gen(x):
    while True:
        yield x


def _open_io(file_path: str | Path | IO, mode: str = 'rb') -> IO:
    if isinstance(file_path, str) or isinstance(file_path, Path):
        return open(file_path, 'rb')
    return file_path


class ResourcesApi(CreatableEntityApi[Resource], DeletableEntityApi[Resource]):
    """API handler for resource-related endpoints."""

    _ENDPOINT_BASE = 'resources'

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None,
                 annotations_api=None,
                 projects_api=None
                 ) -> None:
        """Initialize the resources API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        from .annotations_api import AnnotationsApi
        from .projects_api import ProjectsApi
        super().__init__(config, Resource, ResourcesApi._ENDPOINT_BASE, client)
        nest_asyncio.apply()
        self.annotations_api = AnnotationsApi(
            config, client, resources_api=self) if annotations_api is None else annotations_api
        self.projects_api = projects_api or ProjectsApi(config, client, resources_api=self)

    def get_list(self,
                 status: ResourceStatus | None = None,
                 from_date: date | str | None = None,
                 to_date: date | str | None = None,
                 tags: Sequence[str] | None = None,
                 modality: str | None = None,
                 mimetype: str | None = None,
                 #  return_ids_only: bool = False,
                 order_field: ResourceFields | None = None,
                 order_ascending: bool | None = None,
                 channel: str | None = None,
                 project_name: str | list[str] | None = None,
                 filename: str | None = None,
                 limit: int | None = None
                 ) -> Sequence[Resource]:
        """Get resources with optional filtering.

        Args:
            status: The resource status. Possible values: 'inbox', 'published', 'archived' or None. If None, it will return all resources.
            from_date : The start date (inclusive).
            to_date: The end date (exclusive).
            tags: The tags to filter the resources.
            modality: The modality of the resources.
            mimetype: The mimetype of the resources.
            order_field: The field to order the resources. See :data:`~ResourceFields`.
            order_ascending: Whether to order the resources in ascending order.
            project_name: The project name or a list of project names to filter resources by project.
                If multiple projects are provided, resources will be filtered to include only those belonging to ALL of the specified projects.
        """

        # Convert datetime objects to ISO format
        if from_date:
            if isinstance(from_date, str):
                date.fromisoformat(from_date)
            else:
                from_date = from_date.isoformat()
        if to_date:
            if isinstance(to_date, str):
                date.fromisoformat(to_date)
            else:
                to_date = to_date.isoformat()

        # Prepare the payload
        payload = {
            "from": from_date,
            "to": to_date,
            "status": status if status is not None else "",
            "modality": modality,
            "mimetype": mimetype,
            # "ids": return_ids_only,
            "order_field": order_field,
            "order_by_asc": order_ascending,
            "channel_name": channel,
            "filename": filename,
        }
        # remove nones from payload
        payload = {k: v for k, v in payload.items() if v is not None}
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

        return super().get_list(limit=limit, params=payload)

    def get_annotations(self, resource: str | Resource,
                        annotation_type: AnnotationType | str | None = None) -> Sequence[Annotation]:
        """Get annotations for a specific resource.

        Args:
            resource: The resource ID or Resource instance to fetch annotations for.

        Returns:
            A sequence of Annotation objects associated with the specified resource.
        """
        return self.annotations_api.get_list(resource=resource,
                                             load_ai_segmentations=True,
                                             annotation_type=annotation_type)

    @staticmethod
    def __process_files_parameter(file_path: str | Sequence[str | IO | pydicom.Dataset]
                                  ) -> Sequence[str | IO]:
        """
        Process the file_path parameter to ensure it is a list of file paths or IO objects.
        """
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return [f'{file_path}/{f}' for f in os.listdir(file_path) if os.path.isfile(f'{file_path}/{f}')]

        processed_files = []
        for item in file_path:
            if isinstance(item, pydicom.Dataset):
                processed_files.append(to_bytesio(item, item.filename))
            else:
                processed_files.append(item)
        return processed_files

    def _assemble_dicoms(self, files_path: Sequence[str | IO],
                         progress_bar: bool = False
                         ) -> tuple[Sequence[str | IO], bool, Sequence[int]]:
        """
        Assembles DICOM files into a single file.

        Args:
            files_path: The paths to the DICOM files to assemble.

        Returns:
            A tuple containing:
                - The paths to the assembled DICOM files.
                - A boolean indicating if the assembly was necessary.
                - same length as the output assembled DICOMs, mapping assembled DICOM to original DICOMs.
        """
        dicoms_files_path = []
        other_files_path = []
        dicom_original_idxs = []
        others_original_idxs = []
        for i, f in enumerate(files_path):
            if is_dicom(f):
                dicoms_files_path.append(f)
                dicom_original_idxs.append(i)
            else:
                other_files_path.append(f)
                others_original_idxs.append(i)

        orig_len = len(dicoms_files_path)
        if orig_len == 0:
            _LOGGER.debug("No DICOM files found to assemble.")
            return files_path, False, []
        dicoms_files_path = dicom_utils.assemble_dicoms(dicoms_files_path,
                                                        return_as_IO=True,
                                                        progress_bar=progress_bar)

        new_len = len(dicoms_files_path)
        if new_len != orig_len:
            _LOGGER.info(f"Assembled {new_len} dicom files out of {orig_len} files.")
            mapping_idx = [None] * len(files_path)

            files_path = GeneratorWithLength(itertools.chain(dicoms_files_path, other_files_path),
                                             length=new_len + len(other_files_path))
            assembled = True
            for orig_idx, value in zip(dicom_original_idxs, dicoms_files_path.inverse_mapping_idx):
                mapping_idx[orig_idx] = value
            for i, orig_idx in enumerate(others_original_idxs):
                mapping_idx[orig_idx] = new_len + i
        else:
            assembled = False
            mapping_idx = [i for i in range(len(files_path))]

        return files_path, assembled, mapping_idx

    async def _upload_single_resource_async(self,
                                            file_path: str | IO,
                                            mimetype: str | None = None,
                                            anonymize: bool = False,
                                            anonymize_retain_codes: Sequence[tuple] = [],
                                            tags: list[str] = [],
                                            mung_filename: Sequence[int] | Literal['all'] | None = None,
                                            channel: str | None = None,
                                            session=None,
                                            modality: str | None = None,
                                            publish: bool = False,
                                            metadata_file: str | dict | None = None,
                                            ) -> str:
        if is_io_object(file_path):
            source_filepath = os.path.abspath(os.path.expanduser(file_path.name))
            filename = os.path.basename(source_filepath)
        else:
            source_filepath = os.path.abspath(os.path.expanduser(file_path))
            filename = os.path.basename(source_filepath)

        if session is not None and not isinstance(session, aiohttp.ClientSession):
            raise ValueError("session must be an aiohttp.ClientSession object.")

        if mung_filename is not None:
            file_parts = Path(source_filepath).parts
            if file_parts[0] == os.path.sep:
                file_parts = file_parts[1:]
            if mung_filename == 'all':
                new_filename = '_'.join(file_parts)
            else:
                folder_parts = file_parts[:-1]
                new_filename = '_'.join([folder_parts[i-1] for i in mung_filename if i <= len(folder_parts)])
                new_filename += '_' + file_parts[-1]
            filename = new_filename

        is_a_dicom_file = None
        if not mimetype or mimetype == DEFAULT_MIME_TYPE:
            mimetype_list, ext = guess_typez(file_path, use_magic=False)
            for mime in mimetype_list:
                if mime in NIFTI_MIMES:
                    mimetype = DEFAULT_NIFTI_MIME
                    break
            else:
                if ext == '.nii.gz' or filename.lower().endswith('nii.gz'):
                    mimetype = DEFAULT_NIFTI_MIME
                else:
                    mimetype = mimetype_list[-1] if mimetype_list and mimetype_list[-1] else DEFAULT_MIME_TYPE
        if (not mimetype) or mimetype == DEFAULT_MIME_TYPE:
            msg = f"Could not determine mimetype for file {source_filepath}."
            _LOGGER.warning(msg)
            _USER_LOGGER.warning(msg)

        mimetype = standardize_mimetype(mimetype)

        if is_a_dicom_file == True or is_dicom(file_path):
            if tags is None:
                tags = []
            else:
                tags = list(tags)
            ds = pydicom.dcmread(file_path)
            if anonymize:
                _LOGGER.info(f"Anonymizing {file_path}")
                ds = anonymize_dicom(ds, retain_codes=anonymize_retain_codes)
            lat = dicom_utils.get_dicom_laterality(ds)
            if lat == 'L':
                tags.append("left")
            elif lat == 'R':
                tags.append("right")
            # make the dicom `ds` object a file-like object in order to avoid unnecessary disk writes
            f = to_bytesio(ds, filename)
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
            file_key = 'resource'
            form.add_field('source', 'api')

            form.add_field(file_key, f, filename=filename, content_type=mimetype)
            form.add_field('source_filepath', source_filepath)  # full path to the file
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

            resp_data = await self._make_request_async_json('POST',
                                                            endpoint=self.endpoint_base,
                                                            data=form)
            if 'error' in resp_data:
                raise DatamintException(resp_data['error'])
            _LOGGER.debug(f"Response on uploading {filename}: {resp_data}")
            return resp_data['id']
        except Exception as e:
            if 'filename' in locals():
                _LOGGER.error(f"Error uploading {filename}: {e}")
            else:
                _LOGGER.error(f"Error uploading {file_path}: {e}")
            raise
        finally:
            f.close()

    async def _upload_resources_async(self,
                                      files_path: Sequence[str | IO],
                                      mimetype: str | None = None,
                                      anonymize: bool = False,
                                      anonymize_retain_codes: Sequence[tuple] = [],
                                      on_error: Literal['raise', 'skip'] = 'raise',
                                      tags=None,
                                      mung_filename: Sequence[int] | Literal['all'] | None = None,
                                      channel: str | None = None,
                                      modality: str | None = None,
                                      publish: bool = False,
                                      segmentation_files: Sequence[dict] | None = None,
                                      transpose_segmentation: bool = False,
                                      metadata_files: Sequence[str | dict | None] | None = None,
                                      progress_bar: tqdm | None = None,
                                      ) -> list[str]:
        if on_error not in ['raise', 'skip']:
            raise ValueError("on_error must be either 'raise' or 'skip'")

        if segmentation_files is None:
            segmentation_files = _infinite_gen(None)

        if metadata_files is None:
            metadata_files = _infinite_gen(None)

        async with aiohttp.ClientSession() as session:
            async def __upload_single_resource(file_path, segfiles: dict[str, list | dict],
                                               metadata_file: str | dict | None):
                name = file_path.name if is_io_object(file_path) else file_path
                name = os.path.basename(name)
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
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix(file=name)
                else:
                    _USER_LOGGER.info(f'"{name}" uploaded')

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
                            await self.annotations_api._upload_segmentations_async(
                                rid,
                                file_path=f,
                                name=name,
                                frame_index=frame_index,
                                transpose_segmentation=transpose_segmentation
                            )
                return rid

            try:
                tasks = [__upload_single_resource(f, segfiles, metadata_file)
                         for f, segfiles, metadata_file in zip(files_path, segmentation_files, metadata_files)]
            except ValueError:
                msg = f"Error preparing upload tasks. Try `assemble_dicom=False`."
                _LOGGER.error(msg)
                _USER_LOGGER.error(msg)
                raise
            return await asyncio.gather(*tasks, return_exceptions=on_error == 'skip')

    def upload_resources(self,
                         files_path: Sequence[str | IO | pydicom.Dataset],
                         mimetype: str | None = None,
                         anonymize: bool = False,
                         anonymize_retain_codes: Sequence[tuple] = [],
                         on_error: Literal['raise', 'skip'] = 'raise',
                         tags: Sequence[str] | None = None,
                         mung_filename: Sequence[int] | Literal['all'] | None = None,
                         channel: str | None = None,
                         publish: bool = False,
                         publish_to: str | None = None,
                         segmentation_files: Sequence[Sequence[str] | dict] | None = None,
                         transpose_segmentation: bool = False,
                         modality: str | None = None,
                         assemble_dicoms: bool = True,
                         metadata: Sequence[str | dict | None] | None = None,
                         discard_dicom_reports: bool = True,
                         progress_bar: bool = False
                         ) -> Sequence[str | Exception]:
        """
        Upload multiple resources.

        Note: For uploading a single resource, use `upload_resource()` instead.

        Args:
            files_path: A sequence of paths to resource files, IO objects, or pydicom.Dataset objects.
                Must contain at least 2 items. Supports mixed types within the sequence.
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
            metadata (Optional[list[str | dict | None]]): JSON metadata to include with each resource.
                Must have the same length as `files_path`.
                Can be file paths (str) or already loaded dictionaries (dict).

        Raises:
            ValueError: If a single resource is provided instead of multiple resources.
            ResourceNotFoundError: If `publish_to` is supplied, and the project does not exists.

        Returns:
            list[str | Exception]: A list of resource IDs or errors.
        """

        if on_error not in ['raise', 'skip']:
            raise ValueError("on_error must be either 'raise' or 'skip'")

        # Check if single resource provided and raise error (list of 1 item is allowed)
        if isinstance(files_path, IO) or isinstance(files_path, pydicom.Dataset) or (isinstance(files_path, str) and not os.path.isdir(files_path)):
            raise ValueError(
                "upload_resources() only accepts multiple resources. For single resource upload, use upload_resource() instead.")

        files_path = ResourcesApi.__process_files_parameter(files_path)

        # Discard DICOM reports
        if discard_dicom_reports:
            old_size = len(files_path)
            # Create filtered lists maintaining index correspondence
            filtered_files = []
            filtered_metadata = []

            for i, f in enumerate(files_path):
                if not is_dicom_report(f):
                    filtered_files.append(f)
                    if metadata is not None:
                        filtered_metadata.append(metadata[i])

            files_path = filtered_files
            if metadata is not None:
                metadata = filtered_metadata

            if old_size is not None and old_size != len(files_path):
                _LOGGER.info(f"Discarded {old_size - len(files_path)} DICOM report files from upload.")

        if isinstance(metadata, (str, dict)):
            _LOGGER.debug("Converting metadatas to a list")
            metadata = [metadata]

        if metadata is not None and len(metadata) != len(files_path):
            raise ValueError("The number of metadata files must match the number of resources.")
        if assemble_dicoms:
            files_path, assembled, mapping_idx = self._assemble_dicoms(files_path, progress_bar=progress_bar)
            assemble_dicoms = assembled
        else:
            mapping_idx = [i for i in range(len(files_path))]
        n_files = len(files_path)

        if n_files <= 1:
            # Disable progress bar for single file uploads
            progress_bar = False

        if segmentation_files is not None:
            if assemble_dicoms:
                raise NotImplementedError("Segmentation files cannot be uploaded when assembling dicoms yet.")
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
        pbar = None
        try:
            if progress_bar:
                pbar = tqdm(total=n_files, desc="Uploading resources", unit="file")

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
                                                progress_bar=pbar
                                                )

            resource_ids = loop.run_until_complete(task)
        finally:
            if pbar:
                pbar.close()

        _LOGGER.info(f"Resources uploaded: {resource_ids}")

        if publish_to is not None:
            _USER_LOGGER.info('Adding resources to project')
            resource_ids_succ = [rid for rid in resource_ids if not isinstance(rid, Exception)]
            try:
                self.projects_api.add_resources(resource_ids_succ, publish_to)
            except Exception as e:
                _LOGGER.error(f"Error adding resources to project: {e}")
                if on_error == 'raise':
                    raise e

        if mapping_idx:
            _LOGGER.debug(f"Mapping indices for DICOM files: {mapping_idx}")
            resource_ids = [resource_ids[idx] for idx in mapping_idx]

        return resource_ids

    def upload_resource(self,
                        file_path: str | IO | pydicom.Dataset,
                        mimetype: str | None = None,
                        anonymize: bool = False,
                        anonymize_retain_codes: Sequence[tuple] = [],
                        tags: Sequence[str] | None = None,
                        mung_filename: Sequence[int] | Literal['all'] | None = None,
                        channel: str | None = None,
                        publish: bool = False,
                        publish_to: str | None = None,
                        segmentation_files: dict | None = None,
                        transpose_segmentation: bool = False,
                        modality: str | None = None,
                        metadata: dict | str | None = None,
                        discard_dicom_reports: bool = True
                        ) -> str:
        """
        Upload a single resource.

        This is a convenience method that wraps upload_resources for single file uploads.
        It provides a cleaner interface when uploading just one file.

        Args:
            file_path: The path to the resource file or IO object.
            mimetype: The mimetype of the resource. If None, it will be guessed.
            anonymize: Whether to anonymize the DICOM or not.
            anonymize_retain_codes: The tags to retain when anonymizing the DICOM.
            tags: The tags to add to the resource.
            mung_filename: The parts of the filepath to keep when renaming the resource file.
                'all' keeps all parts.
            channel: The channel to upload the resource to. An arbitrary name to group the resources.
            publish: Whether to directly publish the resource or not. It will have the 'published' status.
            publish_to: The project name or id to publish the resource to.
                It will have the 'published' status and will be added to the project.
                If this is set, `publish` parameter is ignored.
            segmentation_files: The segmentation files to upload. Should be a dict with:
                - 'files': A list of paths to the segmentation files. Example: ['seg1.nii.gz', 'seg2.nii.gz'].
                - 'names': A dict mapping pixel values to class names. Example: {1: 'Brain', 2: 'Lung'}.
            transpose_segmentation: Whether to transpose the segmentation files or not.
            modality: The modality of the resource.
            metadata: JSON metadata to include with the resource.
                Can be a file path (str) or already loaded dictionary (dict).
            discard_dicom_reports: Whether to discard DICOM reports or not.

        Returns:
            str: The resource ID of the uploaded resource.

        Raises:
            ResourceNotFoundError: If `publish_to` is supplied, and the project does not exist.
            DatamintException: If the upload fails.

        Example:
            .. code-block:: python

                # Simple upload
                resource_id = api.resources.upload_resource('path/to/file.dcm')

                # Upload with metadata and segmentation
                resource_id = api.resources.upload_resource(
                    'path/to/file.dcm',
                    tags=['tutorial', 'case1'],
                    channel='study_channel',
                    segmentation_files={
                        'files': ['path/to/segmentation.nii.gz'],
                        'names': {1: 'Brain', 2: 'Lung'}
                    },
                    metadata={'patient_age': 45, 'modality': 'CT'}
                )
        """
        # Convert segmentation_files to the format expected by upload_resources
        segmentation_files_list: list[list[str] | dict] | None = None
        if segmentation_files is not None:
            segmentation_files_list = [segmentation_files]

        # Call upload_resources with single file
        result = self.upload_resources(
            files_path=[file_path],
            mimetype=mimetype,
            anonymize=anonymize,
            anonymize_retain_codes=anonymize_retain_codes,
            tags=tags,
            mung_filename=mung_filename,
            channel=channel,
            publish=publish,
            publish_to=publish_to,
            segmentation_files=segmentation_files_list,
            transpose_segmentation=transpose_segmentation,
            modality=modality,
            metadata=[metadata],
            discard_dicom_reports=discard_dicom_reports,
            assemble_dicoms=False,  # No need to assemble for single file
            progress_bar=False  # Disable progress bar for single uploads
        )

        # upload_resources returns a list, so we extract the first element
        if isinstance(result, Sequence) and len(result) == 1:
            r = result[0]
            if isinstance(r, Exception):
                raise r
            return r
        else:
            # This should not happen with single file uploads, but handle it just in case
            raise DatamintException(f"Unexpected return from upload_resources: {type(result)} | {result}")

    async def _async_download_file(self,
                                   resource: str | Resource,
                                   save_path: str | Path,
                                   session: aiohttp.ClientSession | None = None,
                                   progress_bar: tqdm | None = None,
                                   add_extension: bool = False) -> str:
        """
        Asynchronously download a file from the server.

        Args:
            resource: The resource unique id or Resource object.
            save_path: The path to save the file.
            session: The aiohttp session to use for the request.
            progress_bar: Optional progress bar to update after download completion.
            add_extension: Whether to add the appropriate file extension based on content type.

        Returns:
            str: The actual path where the file was saved (important when add_extension=True).
        """
        save_path = str(save_path)  # Ensure save_path is a string for file operations
        resource_id = self._entid(resource)

        # Disable total timeout for large file downloads, keep connect timeout
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.config.timeout)

        try:
            async with self._make_request_async('GET',
                                                f'{self.endpoint_base}/{resource_id}/file',
                                                session=session,
                                                headers={'accept': 'application/octet-stream'},
                                                timeout=timeout) as resp:

                final_save_path = save_path
                target_path = final_save_path
                if add_extension:
                    target_path = f"{save_path}.tmp"

                with open(target_path, 'wb') as f:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):  # 1MB chunks
                        f.write(chunk)

            if add_extension:
                # Determine mimetype from file content (read first 2KB)
                with open(target_path, 'rb') as f:
                    head_content = f.read(2048)

                mimetype, ext = BaseApi._determine_mimetype(content=head_content,
                                                            declared_mimetype=resource.mimetype if isinstance(resource, Resource) else None)

                # Generate final path with extension if needed
                if mimetype is not None and mimetype != DEFAULT_MIME_TYPE:
                    if ext is None:
                        ext = guess_extension(mimetype)
                    if ext is not None and not save_path.endswith(ext):
                        final_save_path = save_path + ext

                # Move file to final location
                os.rename(target_path, final_save_path)

            if progress_bar:
                progress_bar.update(1)

            return final_save_path

        except ResourceNotFoundError as e:
            e.set_params('resource', {'resource_id': resource_id})
            raise e

    def download_multiple_resources(self,
                                    resources: Sequence[str] | Sequence[Resource],
                                    save_path: Sequence[str] | str,
                                    add_extension: bool = False,
                                    overwrite: bool = True
                                    ) -> list[str]:
        """
        Download multiple resources and save them to the specified paths.
        This is faster than downloading them one by one.

        Args:
            resources: A list of resource unique ids.
            save_path : A list of paths to save the files or a directory path, of same length as resources.
                If a directory path is provided, files will be saved in that directory.
            add_extension: Whether to add the appropriate file extension to the save_path based on the content type.

        Returns:
            list[str]: A list of paths where the files were saved. Important if `add_extension=True`.
        """
        if isinstance(resources, str):
            raise ValueError("resources must be a list of resources")

        async def _download_all_async():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._async_download_file(
                        resource=r,
                        save_path=path,
                        session=session,
                        progress_bar=progress_bar,
                        add_extension=add_extension
                    )
                    for r, path in zip(resources, save_path)
                ]
                return await asyncio.gather(*tasks)

        if isinstance(save_path, str):
            save_path = [os.path.join(save_path, self._entid(r)) for r in resources]

        if len(save_path) != len(resources):
            raise ValueError("The number of save paths must match the number of resources.")

        if not overwrite:
            new_resources = []
            new_save_path = []
            for i in range(len(resources)):
                if not os.path.exists(save_path[i]):
                    new_resources.append(resources[i])
                    new_save_path.append(save_path[i])
            resources = new_resources
            save_path = new_save_path

        with tqdm(total=len(resources), desc="Downloading resources", unit="file") as progress_bar:
            loop = asyncio.get_event_loop()
            final_save_paths = loop.run_until_complete(_download_all_async())

        return final_save_paths

    def download_resource_file(self,
                               resource: str | Resource,
                               save_path: str | None = None,
                               auto_convert: bool = True,
                               add_extension: bool = False
                               ) -> ImagingData | tuple[ImagingData, str] | bytes:
        """
        Download a resource file.

        Args:
            resource: The resource unique id or Resource instance.
            save_path: The path to save the file.
            auto_convert: Whether to convert the file to a known format or not.
            add_extension: Whether to add the appropriate file extension to the save_path based on the content type.

        Returns:
            The resource content in bytes (if `auto_convert=False`) or the resource object (if `auto_convert=True`).
            if `add_extension=True`, the function will return a tuple of (resource_data, save_path).

        Raises:
            ResourceNotFoundError: If the resource does not exists.

        Example:
            >>> api_handler.download_resource_file('resource_id', auto_convert=False)
                returns the resource content in bytes.
            >>> api_handler.download_resource_file('resource_id', auto_convert=True)
                Assuming this resource is a dicom file, it will return a pydicom.Dataset object. 
            >>> api_handler.download_resource_file('resource_id', save_path='path/to/dicomfile.dcm')
                saves the file in the specified path.
        """
        if save_path is None and add_extension:
            raise ValueError("If add_extension is True, save_path must be provided.")

        try:
            response = self._make_entity_request('GET',
                                                 resource,
                                                 add_path='file',
                                                 headers={'accept': 'application/octet-stream'})

            # Get mimetype if needed for auto_convert or add_extension
            mimetype = None
            ext = None
            if auto_convert or add_extension:
                mimetype, ext = BaseApi._determine_mimetype(content=response.content,
                                                            declared_mimetype=resource.mimetype if isinstance(resource, Resource) else None)
            if auto_convert:
                if mimetype is None:
                    _LOGGER.warning("Could not determine mimetype. Returning a bytes array.")
                    resource_file = response.content
                else:
                    try:
                        resource_file = BaseApi.convert_format(response.content,
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
            e.set_params('resource', {'resource_id': self._entid(resource)})
            raise e

        if save_path is not None:
            if add_extension and mimetype is not None:
                if ext is None:
                    ext = guess_extension(mimetype)
                if ext is not None and not save_path.endswith(ext):
                    save_path += ext
            with open(save_path, 'wb') as f:
                f.write(response.content)

            if add_extension:
                return resource_file, save_path
        return resource_file

    def cache_resources(
        self,
        resources: Sequence[Resource],
        progress_bar: bool = True,
    ) -> None:
        """Cache multiple resources in parallel, skipping already cached ones.

        This method downloads and caches resource file data concurrently,
        significantly improving efficiency when working with large datasets.
        Only resources that are not already cached will be downloaded.

        Args:
            resources: Sequence of Resource instances to cache.
            progress_bar: Whether to show a progress bar. Default is True.

        Example:
            >>> resources = api.resources.get_list(limit=100)
            >>> api.resources.cache_resources(resources)
            Caching resources: 100%|██████████| 85/85 [00:12<00:00,  6.8files/s]
        """
        # Filter out already cached resources
        resources_to_cache = [res for res in resources if not res.is_cached()]

        if not resources_to_cache:
            _LOGGER.info("All resources are already cached.")
            return

        _LOGGER.info(f"Caching {len(resources_to_cache)} of {len(resources)} resources...")

        if progress_bar:
            pbar = tqdm(total=len(resources_to_cache), desc="Caching resources", unit="file")
        else:
            pbar = None

        try:
            for res in resources_to_cache:
                res._api = self  # Ensure the resource has a reference to the API
                res.fetch_file_data(auto_convert=False, use_cache=True)
                if pbar:
                    pbar.set_postfix(filename=res.filename)
                    pbar.update(1)

        finally:
            if pbar:
                pbar.close()
        _LOGGER.info(f"Successfully cached {len(resources_to_cache)} resources.")

    def download_resource_frame(self,
                                resource: str | Resource,
                                frame_index: int) -> Image.Image:
        """
        Download a frame of a resource.
        This is faster than downloading the whole resource and then extracting the frame.

        Args:
            resource: The resource unique id or Resource object.
            frame_index: The index of the frame to download.

        Returns:
            Image.Image: The frame as a PIL image.

        Raises:
            ResourceNotFoundError: If the resource does not exists.
            DatamintException: If the resource is not a video or dicom.
        """
        # check if the resource is an single frame image (png,jpeg,...) first.
        # If so, download the whole resource file and return the image.
        if not isinstance(resource, Resource):
            resource = self.get_by_id(resource)
        if resource.mimetype.startswith('image/') or resource.storage == 'ImageResource':
            if frame_index != 0:
                raise DatamintException(f"Resource {resource.id} is not a multi-frame resource, "
                                        f"but frame_index is {frame_index}.")
            return self.download_resource_file(resource, auto_convert=True)

        try:
            response = self._make_entity_request('GET',
                                                 resource,
                                                 add_path=f'frames/{frame_index}',
                                                 headers={'accept': 'image/*'})
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            else:
                raise DatamintException(
                    f"Error downloading frame {frame_index} of resource {self._entid(resource)}: {response.text}")
        except ResourceNotFoundError as e:
            e.set_params('resource', {'resource_id': self._entid(resource)})
            raise e

    def publish_resources(self,
                          resources: str | Resource | Sequence[str | Resource]) -> None:
        """
        Publish resources, changing their status to 'published'.

        Args:
            resources: The resources to publish. Can be a Resource object (instead of a list)

        Raises:
            ResourceNotFoundError: If the resource does not exists or the project does not exists.
        """
        if isinstance(resources, (Resource, str)):
            resources = [resources]

        for resource in resources:
            try:
                self._make_entity_request('POST', resource, add_path='publish')
            except ResourceNotFoundError as e:
                e.set_params('resource', {'resource_id': self._entid(resource)})
                raise
            except httpx.HTTPError as e:
                if BaseApi._has_status_code(e, 400) and 'Resource must be in inbox status to be approved' in e.response.text:
                    _LOGGER.warning(f"Resource {resource} is not in inbox status. Skipping publishing")
                else:
                    raise

    def set_tags(self,
                 resource: str | Resource | Sequence[str | Resource],
                 tags: Sequence[str],
                 ):
        """
        Set tags for a resource, IMPORTANT: This replaces all existing tags.
        Args:
            resource: The resource object or a list of resources.
            tags: The tags to set.
        """

        uniq_tags = set(tags)  # remove duplicates

        if isinstance(resource, Sequence):
            resource_ids = [self._entid(res) for res in resource]
            response = self._make_request('PUT',
                                          f'{self.endpoint_base}/tags',
                                          json={'resource_ids': resource_ids,
                                                'tags': list(uniq_tags)})
        else:
            resource_id = self._entid(resource)
            response = self._make_entity_request('PUT',
                                                 resource_id,
                                                 add_path='tags',
                                                 json={'tags': list(uniq_tags)})
        return response

    # def get_projects(self, resource: Resource) -> Sequence[Project]:
    #     """
    #     Get all projects this resource belongs to.

    #     Args:
    #         resource: The Resource instance.

    #     Returns:
    #         List of Project instances
    #     """
    #     resource._ensure_attr('projects')
    #     proj_ids = [p['id'] for p in resource.projects]
    #     return [proj for proj in self.projects_api.get_all() if proj.id in proj_ids]

    def add_tags(self,
                 resource: str | Resource | Sequence[str | Resource],
                 tags: Sequence[str],
                 ):
        """
        Add tags to a resource, IMPORTANT: This appends to existing tags.
        Args:
            resource: The resource unique id or Resource object.
            tags: The tags to add.
        """
        if isinstance(resource, str):
            resource = self.get_by_id(resource)
        elif isinstance(resource, Sequence):
            # Transform every str to Resource first.
            resources = [self.get_by_id(res) if isinstance(res, str) else res for res in resource]

            # group resource having the exact same tags to minimize requests
            tag_map: dict[tuple, list[Resource]] = defaultdict(list)
            for res in resources:
                old_tags = res.tags if res.tags is not None else []
                # key = tuple(sorted(old_tags))
                key = tuple(old_tags)  # keep order, assuming order matters for tags
                tag_map[key].append(res)

            # finally, set tags for each group
            for old_tags_tuple, res_group in tag_map.items():
                old_tags = list(old_tags_tuple)
                self.set_tags(res_group, old_tags + list(tags))
            return

        old_tags = resource.tags if resource.tags is not None else []
        self.set_tags(resource, old_tags + list(tags))

    def bulk_delete(self, entities: Sequence[str | Resource]) -> None:
        """Delete multiple entities. Faster than deleting them one by one.

        Args:
            entities: Sequence of unique identifiers for the entities to delete or the entity instances themselves.

        Raises:
            httpx.HTTPStatusError: If deletion fails or any entity not found
        """
        from math import ceil

        resources_ids = [self._entid(ent) for ent in entities]
        if len(resources_ids) == 0:
            return
        batch_size = 200
        for i in range(0, ceil(len(resources_ids)/batch_size)):
            batch_ids = resources_ids[i*batch_size:(i+1)*batch_size]
            self._make_request('DELETE',
                               f'{self.endpoint_base}',
                               params={'resource_ids': ','.join(batch_ids)})
