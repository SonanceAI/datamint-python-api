from typing import Literal, BinaryIO, IO, Any, overload
from collections.abc import Sequence, Generator
import httpx
from datetime import date
import logging
from ..entity_base_api import ApiConfig, CreatableEntityApi, DeletableEntityApi
from datamint.entities.annotations.annotation import Annotation
from datamint.entities.resource import Resource
from datamint.api.dto import AnnotationType, CreateAnnotationDto, LineGeometry, BoxGeometry, CoordinateSystem, Geometry
import numpy as np
import os
import aiohttp
import json
from datamint.exceptions import DatamintException, ResourceNotFoundError
from medimgkit.nifti_utils import DEFAULT_NIFTI_MIME
from medimgkit.format_detection import guess_type
import nibabel as nib
from PIL import Image
from io import BytesIO
import pydicom
from pathlib import Path
from tqdm.auto import tqdm
import asyncio

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')
MAX_NUMBER_DISTINCT_COLORS = 2048  # Maximum number of distinct colors in a segmentation image


class AnnotationsApi(CreatableEntityApi[Annotation], DeletableEntityApi[Annotation]):
    """API handler for annotation-related endpoints."""

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None,
                 models_api=None,
                 resources_api=None) -> None:
        """Initialize the annotations API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        from .resources_api import ResourcesApi
        from .models_api import ModelsApi

        super().__init__(config, Annotation, 'annotations', client)
        self._models_api = ModelsApi(config, client=client) if models_api is None else models_api
        self._resources_api = ResourcesApi(
            config, client=client, annotations_api=self) if resources_api is None else resources_api

    @overload
    def get_list(self,
                 resource: str | Resource | Sequence[str | Resource] | None = None,
                 annotation_type: AnnotationType | str | None = None,
                 annotator_email: str | None = None,
                 date_from: date | None = None,
                 date_to: date | None = None,
                 dataset_id: str | None = None,
                 worklist_id: str | None = None,
                 status: Literal['new', 'published'] | None = None,
                 load_ai_segmentations: bool | None = None,
                 limit: int | None = None,
                 group_by_resource: Literal[False] = False
                 ) -> Sequence[Annotation]: ...

    @overload
    def get_list(self,
                 resource: str | Resource | Sequence[str | Resource] | None = None,
                 annotation_type: AnnotationType | str | None = None,
                 annotator_email: str | None = None,
                 date_from: date | None = None,
                 date_to: date | None = None,
                 dataset_id: str | None = None,
                 worklist_id: str | None = None,
                 status: Literal['new', 'published'] | None = None,
                 load_ai_segmentations: bool | None = None,
                 limit: int | None = None,
                 *,
                 group_by_resource: Literal[True]
                 ) -> Sequence[Sequence[Annotation]]: ...

    def get_list(self,
                 resource: str | Resource | Sequence[str | Resource] | None = None,
                 annotation_type: AnnotationType | str | None = None,
                 annotator_email: str | None = None,
                 date_from: date | None = None,
                 date_to: date | None = None,
                 dataset_id: str | None = None,
                 worklist_id: str | None = None,
                 status: Literal['new', 'published'] | None = None,
                 load_ai_segmentations: bool | None = None,
                 limit: int | None = None,
                 group_by_resource: bool = False
                 ) -> Sequence[Annotation] | Sequence[Sequence[Annotation]]:
        """
        Retrieve a list of annotations with optional filtering.

        Args:
            resource: The resource unique id(s) or Resource instance(s). Can be a single resource,
                a list of resources, or None to retrieve annotations from all resources.
            annotation_type: Filter by annotation type (e.g., 'segmentation', 'category').
            annotator_email: Filter by annotator email address.
            date_from: Filter annotations created on or after this date.
            date_to: Filter annotations created on or before this date.
            dataset_id: Filter by dataset unique id.
            worklist_id: Filter by annotation worklist unique id.
            status: Filter by annotation status ('new' or 'published').
            load_ai_segmentations: Whether to load AI-generated segmentations.
            limit: Maximum number of annotations to return.
            group_by_resource: If True, return results grouped by resource.
                For instance, the first index of the returned list will contain all annotations for the first resource.

        Returns:
            Sequence[Annotation] | Sequence[Sequence[Annotation]]: List of annotations, or list of lists if grouped by resource.

        Example:
            .. code-block:: python

                # Get all annotations for a single resource
                annotations = api.annotations.get_list(resource='resource_id')

                # Get annotations with filters
                annotations = api.annotations.get_list(
                    resource='resource_id',
                    annotation_type='segmentation',
                    status='published'
                )

                # Get annotations for multiple resources
                annotations = api.annotations.get_list(
                    resource=['resource_id_1', 'resource_id_2', 'resource_id_3']
                )
        """
        def group_annotations_by_resource(annotations: Sequence[Annotation],
                                          resource_ids: Sequence[str]
                                          ) -> Sequence[Sequence[Annotation]]:
            resource_annotations_map = {rid: [] for rid in resource_ids}
            for ann in annotations:
                resource_annotations_map[ann.resource_id].append(ann)
            return [resource_annotations_map[rid] for rid in resource_ids]

        # Build search payload according to POST /annotations/search schema
        payload = {
            'annotation_type': annotation_type,
            'annotatorEmail': annotator_email,
            'from': date_from.isoformat() if date_from is not None else None,
            'to': date_to.isoformat() if date_to is not None else None,
            'dataset_id': dataset_id,
            'annotation_worklist_id': worklist_id,
            'status': status,
            'load_ai_segmentations': load_ai_segmentations,
        }

        if isinstance(resource, (str, Resource)):
            resource_id = self._entid(resource)
            payload['resource_id'] = resource_id
            resource_ids = None
        elif resource is not None:
            resource_ids = [self._entid(res) for res in resource]
            payload['resource_ids'] = resource_ids
        else:
            resource_ids = None

        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        items_gen = self._make_request_with_pagination('POST',
                                                       f'{self.endpoint_base}/search',
                                                       return_field=self.endpoint_base,
                                                       limit=limit,
                                                       json=payload)

        all_items = []
        for _, items in items_gen:
            all_items.extend(items)

        all_annotations = [self._init_entity_obj(**item) for item in all_items]

        if group_by_resource and resource_ids is not None:
            return group_annotations_by_resource(all_annotations, resource_ids)
        return all_annotations

    async def _upload_segmentations_async(self,
                                          resource: str | Resource,
                                          frame_index: int | Sequence[int] | None,
                                          file_path: str | np.ndarray,
                                          name: dict[int, str] | dict[tuple, str],
                                          imported_from: str | None = None,
                                          author_email: str | None = None,
                                          discard_empty_segmentations: bool = True,
                                          worklist_id: str | None = None,
                                          model_id: str | None = None,
                                          transpose_segmentation: bool = False,
                                          upload_volume: bool | str = 'auto',
                                          ) -> Sequence[str]:
        """
        Upload segmentations asynchronously.

        Args:
            resource: The resource unique id or Resource instance.
            frame_index: The frame index or None for multiple frames.
            file_path: Path to segmentation file or numpy array.
            name: The name of the segmentation or mapping of pixel values to names.
            imported_from: The imported from value.
            author_email: The author email.
            discard_empty_segmentations: Whether to discard empty segmentations.
            worklist_id: The annotation worklist unique id.
            model_id: The model unique id.
            transpose_segmentation: Whether to transpose the segmentation.
            upload_volume: Whether to upload the volume as a single file or split into frames.

        Returns:
            List of annotation IDs created.
        """
        if upload_volume == 'auto':
            if isinstance(file_path, str) and (file_path.endswith('.nii') or file_path.endswith('.nii.gz')):
                upload_volume = True
            else:
                upload_volume = False

        resource_id = self._entid(resource)
        # Handle volume upload
        if upload_volume:
            if frame_index is not None:
                _LOGGER.warning("frame_index parameter ignored when upload_volume=True")

            return await self._upload_volume_segmentation_async(
                resource_id=resource_id,
                file_path=file_path,
                name=name,
                imported_from=imported_from,
                author_email=author_email,
                worklist_id=worklist_id,
                model_id=model_id,
                transpose_segmentation=transpose_segmentation
            )

        # Handle frame-by-frame upload (existing logic)
        nframes, fios = AnnotationsApi._generate_segmentations_ios(
            file_path, transpose_segmentation=transpose_segmentation
        )
        if frame_index is None:
            frames_indices = list(range(nframes))
        elif isinstance(frame_index, int):
            frames_indices = [frame_index]
        elif isinstance(frame_index, Sequence):
            if len(frame_index) != nframes:
                raise ValueError("Length of frame_index does not match number of frames in segmentation.")
            frames_indices = list(frame_index)
        else:
            raise ValueError("frame_index must be a list of integers or None.")

        annotids = []
        for fidx, f in zip(frames_indices, fios):
            frame_annotids = await self._upload_single_frame_segmentation_async(
                resource_id=resource_id,
                frame_index=fidx,
                fio=f,
                name=name,
                imported_from=imported_from,
                author_email=author_email,
                discard_empty_segmentations=discard_empty_segmentations,
                worklist_id=worklist_id,
                model_id=model_id
            )
            annotids.extend(frame_annotids)
        return annotids

    async def _upload_single_frame_segmentation_async(self,
                                                      resource_id: str,
                                                      frame_index: int | None,
                                                      fio: IO,
                                                      name: dict[int, str] | dict[tuple, str],
                                                      imported_from: str | None = None,
                                                      author_email: str | None = None,
                                                      discard_empty_segmentations: bool = True,
                                                      worklist_id: str | None = None,
                                                      model_id: str | None = None
                                                      ) -> list[str]:
        """
        Upload a single frame segmentation asynchronously.

        Args:
            resource_id: The resource unique id.
            frame_index: The frame index for the segmentation.
            fio: File-like object containing the segmentation image.
            name: The name of the segmentation, a dictionary mapping pixel values to names,
                  or a dictionary mapping RGB tuples to names.
            imported_from: The imported from value.
            author_email: The author email.
            discard_empty_segmentations: Whether to discard empty segmentations.
            worklist_id: The annotation worklist unique id.
            model_id: The model unique id.

        Returns:
            List of annotation IDs created.
        """
        try:
            try:
                img_pil = Image.open(fio)
                img_array = np.array(img_pil)  # shape: (height, width, channels)
                # Returns a list of (count, color) tuples
                unique_vals = img_pil.getcolors(maxcolors=MAX_NUMBER_DISTINCT_COLORS)
                # convert to list of RGB tuples
                if unique_vals is None:
                    raise ValueError(f'Number of unique colors exceeds {MAX_NUMBER_DISTINCT_COLORS}.')
                unique_vals = [color for count, color in unique_vals]
                # Remove black/transparent pixels
                black_pixel = (0, 0, 0)
                unique_vals = [rgb for rgb in unique_vals if rgb != black_pixel]

                if discard_empty_segmentations:
                    if len(unique_vals) == 0:
                        msg = f"Discarding empty RGB segmentation for frame {frame_index}"
                        _LOGGER.debug(msg)
                        _USER_LOGGER.debug(msg)
                        return []
                segnames = AnnotationsApi._get_segmentation_names_rgb(unique_vals, names=name)
                segs_generator = AnnotationsApi._split_rgb_segmentations(img_array, unique_vals)

                fio.seek(0)
                # TODO: Optimize this. It is not necessary to open the image twice.

                # Create annotations
                annotations: list[CreateAnnotationDto] = []
                for segname in segnames:
                    ann = CreateAnnotationDto(
                        type='segmentation',
                        identifier=segname,
                        scope='frame',
                        frame_index=frame_index,
                        imported_from=imported_from,
                        import_author=author_email,
                        model_id=model_id,
                        annotation_worklist_id=worklist_id
                    )
                    annotations.append(ann)

                # Validate unique identifiers
                if len(annotations) != len(set([a.identifier for a in annotations])):
                    raise ValueError(
                        "Multiple annotations with the same identifier, frame_index, scope and author is not supported yet."
                    )

                annotids = await self._create_async(resource_id=resource_id, annotations_dto=annotations)

                # Upload segmentation files
                if len(annotids) != len(segnames):
                    _LOGGER.warning(f"Number of uploaded annotations ({len(annotids)})" +
                                    f" does not match the number of annotations ({len(segnames)})")

                for annotid, segname, fio_seg in zip(annotids, segnames, segs_generator):
                    await self.upload_annotation_file_async(resource_id, annotid, fio_seg,
                                                            content_type='image/png',
                                                            filename=segname)
                return annotids
            finally:
                fio.close()
        except ResourceNotFoundError:
            raise ResourceNotFoundError('resource', {'resource_id': resource_id})

    def _prepare_upload_file(self,
                             file: str | IO,
                             filename: str | None = None,
                             content_type: str | None = None
                             ) -> tuple[IO, str, bool, str | None]:
        if isinstance(file, str):
            if filename is None:
                filename = os.path.basename(file)
            f = open(file, 'rb')
            close_file = True
        else:
            f = file
            if filename is None:
                if hasattr(f, 'name') and isinstance(f.name, str):
                    filename = f.name
                else:
                    filename = 'unnamed_file'
            close_file = False

        if content_type is None:
            content_type, _ = guess_type(filename, use_magic=False)

        return f, filename, close_file, content_type

    async def upload_annotation_file_async(self,
                                           resource: str | Resource,
                                           annotation_id: str,
                                           file: str | IO,
                                           content_type: str | None = None,
                                           filename: str | None = None
                                           ):
        """
        Upload a file for an existing annotation asynchronously.

        Args:
            resource: The resource unique id or Resource instance.
            annotation_id: The annotation unique id.
            file: Path to the file or a file-like object.
            content_type: The MIME type of the file.
            filename: Optional filename to use in the upload. If None and file is a path,
                      the basename of the path will be used.

        Raises:
            DatamintException: If the upload fails.

        Example:
            .. code-block:: python

                await ann_api.upload_annotation_file_async(
                    resource='your_resource_id',
                    annotation_id='your_annotation_id',
                    file='path/to/your/file.png',
                    content_type='image/png',
                    filename='custom_name.png'
                )
        """
        f, filename, close_file, content_type = self._prepare_upload_file(file,
                                                                          filename,
                                                                          content_type=content_type)

        try:
            form = aiohttp.FormData()
            form.add_field('file', f, filename=filename, content_type=content_type)
            resource_id = self._entid(resource)
            endpoint = f'{self.endpoint_base}/{resource_id}/annotations/{annotation_id}/file'
            respdata = await self._make_request_async_json('POST',
                                                           endpoint=endpoint,
                                                           data=form)
            if isinstance(respdata, dict) and 'error' in respdata:
                raise DatamintException(respdata['error'])
        finally:
            if close_file:
                f.close()

    def upload_annotation_file(self,
                               resource: str | Resource,
                               annotation_id: str,
                               file: str | IO,
                               content_type: str | None = None,
                               filename: str | None = None
                               ):
        """
        Upload a file for an existing annotation.

        Args:
            resource: The resource unique id or Resource instance.
            annotation_id: The annotation unique id.
            file: Path to the file or a file-like object.
            content_type: The MIME type of the file.
            filename: Optional filename to use in the upload. If None and file is a path,
                      the basename of the path will be used.

        Raises:
            DatamintException: If the upload fails.
        """
        f, filename, close_file, content_type = self._prepare_upload_file(file,
                                                                          filename,
                                                                          content_type=content_type)
        try:
            files = {
                'file': (filename, f, content_type)
            }
            resource_id = self._entid(resource)
            resp = self._make_request(method='POST',
                                      endpoint=f'{self.endpoint_base}/{resource_id}/annotations/{annotation_id}/file',
                                      files=files)
            respdata = resp.json()
            if isinstance(respdata, dict) and 'error' in respdata:
                raise DatamintException(respdata['error'])
        finally:
            if close_file:
                f.close()

    def create(self,
               resource: str | Resource,
               annotation_dto: CreateAnnotationDto | Sequence[CreateAnnotationDto]
               ) -> str | Sequence[str]:
        """Create one or more annotations for a resource.

        .. warning::
            This is an internal method and should not be used directly by users.
            Please use specific annotation creation methods like 
            :py:meth:`create_image_classification` or :py:meth:`upload_segmentations` instead.

        Args:
            resource (str | Resource): The resource unique id or Resource instance.
            annotation_dto (CreateAnnotationDto | Sequence[CreateAnnotationDto]): 
                A CreateAnnotationDto instance or a list of such instances to be created.

        Returns:
            str | Sequence[str]: The id of the created annotation if a single annotation 
            was provided, or a list of ids if multiple annotations were created.
        """

        annotations = [annotation_dto] if isinstance(annotation_dto, CreateAnnotationDto) else annotation_dto
        annotations = [ann.to_dict() if isinstance(ann, CreateAnnotationDto) else ann for ann in annotations]
        resource_id = self._entid(resource)
        respdata = self._make_request('POST',
                                      f'{self.endpoint_base}/{resource_id}/annotations',
                                      json=annotations).json()
        for r in respdata:
            if isinstance(r, dict) and 'error' in r:
                raise DatamintException(r['error'])
        if isinstance(annotation_dto, CreateAnnotationDto):
            return respdata[0]
        return respdata

    def upload_segmentations(self,
                             resource: str | Resource,
                             file_path: str | Path | np.ndarray,
                             name: str | dict[int, str] | dict[tuple, str] | None = None,
                             frame_index: int | list[int] | None = None,
                             imported_from: str | None = None,
                             author_email: str | None = None,
                             discard_empty_segmentations: bool = True,
                             worklist_id: str | None = None,
                             model_id: str | None = None,
                             transpose_segmentation: bool = False,
                             ai_model_name: str | None = None
                             ) -> list[str]:
        """
        Upload segmentations to a resource.

        Args:
            resource: The resource unique ID or Resource instance.
            file_path: The path to the segmentation file or a numpy array.
                If a numpy array is provided, it can have the shape:
                - (height, width, #frames) or (height, width) for grayscale segmentations
                - (3, height, width, #frames) for RGB segmentations
                For NIfTI files (.nii/.nii.gz), the entire volume is uploaded as a single segmentation.
            name: The name of the segmentation.
                Can be:
                - str: Single name for all segmentations
                - dict[int, str]: Mapping pixel values to names for grayscale segmentations
                - dict[tuple[int, int, int], str]: Mapping RGB tuples to names for RGB segmentations
                Use 'default' as a key for a unnamed classes.
                Example: {(255, 0, 0): 'Red_Region', (0, 255, 0): 'Green_Region'}
            frame_index: The frame index of the segmentation.
                If a list, it must have the same length as the number of frames in the segmentation.
                If None, it is assumed that the segmentations are in sequential order starting from 0.
                This parameter is ignored for NIfTI files as they are treated as volume segmentations.
            imported_from: The imported from value.
            author_email: The author email.
            discard_empty_segmentations: Whether to discard empty segmentations or not.
            worklist_id: The annotation worklist unique id.
            model_id: The model unique id.
            transpose_segmentation: Whether to transpose the segmentation or not.
            ai_model_name: Optional AI model name to associate with the segmentation.

        Returns:
            List of segmentation unique ids.

        Raises:
            ResourceNotFoundError: If the resource does not exist or the segmentation is invalid.
            FileNotFoundError: If the file path does not exist.
            ValueError: If frame_index is provided for NIfTI files or invalid parameters.

        Example:
            .. code-block:: python

                # Grayscale segmentation
                api.annotations.upload_segmentations(resource_id, 'path/to/segmentation.png', 'SegmentationName')

                # RGB segmentation with numpy array
                seg_data = np.random.randint(0, 3, size=(3, 2140, 1760, 1), dtype=np.uint8)
                rgb_names = {(1, 0, 0): 'Red_Region', (0, 1, 0): 'Green_Region', (0, 0, 1): 'Blue_Region'}
                api.annotations.upload_segmentations(resource_id, seg_data, rgb_names)

                # Volume segmentation
                api.annotations.upload_segmentations(resource_id, 'path/to/segmentation.nii.gz', 'VolumeSegmentation')
        """
        import nest_asyncio

        if isinstance(file_path, Path):
            file_path = str(file_path)

        if isinstance(file_path, str) and not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        if ai_model_name is not None:
            model_id = self._models_api.get_by_name(ai_model_name)
            if model_id is None:
                try:
                    available_models = [model['name'] for model in self._models_api.get_all()]
                except Exception:
                    _LOGGER.warning("Could not fetch available AI models from the server.")
                    raise ValueError(f"AI model with name '{ai_model_name}' not found. ")
                raise ValueError(f"AI model with name '{ai_model_name}' not found. " +
                                 f"Available models: {available_models}")
            model_id = model_id['name']

        # Handle NIfTI files specially - upload as single volume
        if isinstance(file_path, str) and (file_path.endswith('.nii') or file_path.endswith('.nii.gz')):
            _LOGGER.info(f"Uploading NIfTI segmentation file: {file_path}")
            if frame_index is not None:
                raise ValueError("Do not provide frame_index for NIfTI segmentations.")

            # Ensure nest_asyncio is applied for Jupyter compatibility
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            task = self._upload_segmentations_async(
                resource=resource,
                frame_index=None,
                file_path=file_path,
                name=name,
                imported_from=imported_from,
                author_email=author_email,
                worklist_id=worklist_id,
                model_id=model_id,
                transpose_segmentation=transpose_segmentation,
                upload_volume=True,
            )
            return loop.run_until_complete(task)

        # All other file types are converted to multiple PNGs and uploaded frame by frame
        standardized_name = self.standardize_segmentation_names(name)
        _LOGGER.debug(f"Standardized segmentation names: {standardized_name}")

        # Handle frame_index parameter
        if isinstance(frame_index, list):
            if len(set(frame_index)) != len(frame_index):
                raise ValueError("frame_index list contains duplicate values.")

        if isinstance(frame_index, Sequence) and len(frame_index) == 1:
            frame_index = frame_index[0]

        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        task = self._upload_segmentations_async(
            resource=resource,
            frame_index=frame_index,
            file_path=file_path,
            name=standardized_name,
            imported_from=imported_from,
            author_email=author_email,
            discard_empty_segmentations=discard_empty_segmentations,
            worklist_id=worklist_id,
            model_id=model_id,
            transpose_segmentation=transpose_segmentation,
            upload_volume=False
        )
        return loop.run_until_complete(task)

    @staticmethod
    def standardize_segmentation_names(name: str | dict | None
                                       ) -> dict:
        """
        Standardize segmentation names to a consistent format.

        Args:
            name: The name input in various formats.

        Returns:
            Standardized name dictionary.
        """
        if name is None:
            return {'default': 'default'}  # Return a dict with integer key for compatibility
        elif isinstance(name, str):
            return {'default': name}  # Use integer key for single string names
        elif isinstance(name, dict):
            # Return the dict as-is since it's already in the correct format
            return name
        else:
            raise ValueError("Invalid name format. Must be str, dict[int, str], dict[tuple, str], or None.")

    async def _create_async(self,
                            resource_id: str,
                            annotations_dto: list[CreateAnnotationDto] | list[dict]) -> list[str]:
        annotations = [ann.to_dict() if isinstance(ann, CreateAnnotationDto) else ann for ann in annotations_dto]
        respdata = await self._make_request_async_json('POST',
                                                       f'{self.endpoint_base}/{resource_id}/annotations',
                                                       json=annotations)
        for r in respdata:
            if isinstance(r, dict) and 'error' in r:
                raise DatamintException(r['error'])
        return respdata

    @staticmethod
    def _get_segmentation_names_rgb(uniq_rgb_vals: list[tuple[int, int, int]],
                                    names: dict[tuple[int, int, int], str]
                                    ) -> list[str]:
        """
        Generate segmentation names for RGB combinations.

        Args:
            uniq_rgb_vals: List of unique RGB combinations as (R,G,B) tuples
            names: Name mapping for RGB combinations

        Returns:
            List of segmentation names
        """
        result = []
        for rgb_tuple in uniq_rgb_vals:
            seg_name = names.get(rgb_tuple, names.get('default', f'seg_{"_".join(map(str, rgb_tuple))}'))
            if seg_name is None:
                if rgb_tuple[0] == rgb_tuple[1] and rgb_tuple[1] == rgb_tuple[2]:
                    msg = f"Provide a name for {rgb_tuple} or {rgb_tuple[0]} or use 'default' key."
                else:
                    msg = f"Provide a name for {rgb_tuple} or use 'default' key."
                raise ValueError(f"RGB combination {rgb_tuple} not found in names dictionary. " +
                                 msg)
            # If using default prefix, append RGB values
            # if rgb_tuple not in names and 'default' in names:
            #     seg_name = f"{seg_name}_{'_'.join(map(str, rgb_tuple))}"
            result.append(seg_name)
        return result

    @staticmethod
    def _split_rgb_segmentations(img: np.ndarray,
                                 uniq_rgb_vals: list[tuple[int, int, int]]
                                 ) -> Generator[BytesIO, None, None]:
        """
        Split RGB segmentations into individual binary masks.

        Args:
            img: RGB image array of shape (height, width, channels)
            uniq_rgb_vals: List of unique RGB combinations as (R,G,B) tuples

        Yields:
            BytesIO objects containing individual segmentation masks
        """
        for rgb_tuple in uniq_rgb_vals:
            # Create binary mask for this RGB combination
            rgb_array = np.array(rgb_tuple[:3])  # Ensure only R,G,B values
            mask = np.all(img[:, :, :3] == rgb_array, axis=2)

            # Convert to uint8 and create PNG
            mask_img = (mask * 255).astype(np.uint8)

            f_out = BytesIO()
            Image.fromarray(mask_img).convert('L').save(f_out, format='PNG')
            f_out.seek(0)
            yield f_out

    async def _upload_volume_segmentation_async(self,
                                                resource_id: str,
                                                file_path: str | np.ndarray,
                                                name: str | dict[int, str] | dict[tuple, str] | None,
                                                imported_from: str | None = None,
                                                author_email: str | None = None,
                                                worklist_id: str | None = None,
                                                model_id: str | None = None,
                                                transpose_segmentation: bool = False
                                                ) -> Sequence[str]:
        """
        Upload a volume segmentation as a single file asynchronously.

        Args:
            resource_id: The resource unique id.
            file_path: Path to segmentation file or numpy array.
            name: The name of the segmentation (string only for volumes).
            imported_from: The imported from value.
            author_email: The author email.
            worklist_id: The annotation worklist unique id.
            model_id: The model unique id.
            transpose_segmentation: Whether to transpose the segmentation.

        Returns:
            List of annotation IDs created.

        Raises:
            ValueError: If name is not a string or file format is unsupported for volume upload.
        """

        if isinstance(name, str):
            raise NotImplementedError("`name=string` is not supported yet for volume segmentation.")
        if isinstance(name, dict):
            if any(isinstance(k, tuple) for k in name.keys()):
                raise NotImplementedError(
                    "For volume segmentations, `name` must be a dictionary with integer keys only.")
            if 'default' in name:
                _LOGGER.warning("Ignoring 'default' key in name dictionary for volume segmentation. Not supported yet.")

        # Prepare file for upload
        if isinstance(file_path, str):
            if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
                # Upload NIfTI file directly
                with open(file_path, 'rb') as f:
                    filename = os.path.basename(file_path)
                    form = aiohttp.FormData()
                    form.add_field('file', f, filename=filename, content_type=DEFAULT_NIFTI_MIME)
                    if model_id is not None:
                        form.add_field('model_id', model_id)  # Add model_id if provided
                    if worklist_id is not None:
                        form.add_field('annotation_worklist_id', worklist_id)
                    if name is not None:
                        form.add_field('segmentation_map', json.dumps(name), content_type='application/json')

                    try:
                        respdata = await self._make_request_async_json(
                            'POST',
                            f'{self.endpoint_base}/{resource_id}/segmentations/file',
                            data=form
                        )
                    except ResourceNotFoundError as e:
                        e.resource_type = 'resource'
                        e.params = {'resource_id': resource_id}
                        raise e

                    if 'error' in respdata:
                        raise DatamintException(respdata['error'])
                    return respdata
            else:
                raise ValueError(f"Volume upload not supported for file format: {file_path}")
        elif isinstance(file_path, np.ndarray):
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported file_path type for volume upload: {type(file_path)}")

        _USER_LOGGER.info(f'Volume segmentation uploaded for resource {resource_id}')

    @staticmethod
    def _generate_segmentations_ios(file_path: str | np.ndarray,
                                    transpose_segmentation: bool = False
                                    ) -> tuple[int, Generator[BinaryIO, None, None]]:
        if not isinstance(file_path, (str, np.ndarray)):
            raise ValueError(f"Unsupported file type: {type(file_path)}")

        if isinstance(file_path, np.ndarray):
            normalized_imgs = AnnotationsApi._normalize_segmentation_array(file_path)
            # normalized_imgs shape: (3, height, width, #frames)

            # Apply transpose if requested
            if transpose_segmentation:
                # (channels, height, width, frames) -> (channels, width, height, frames)
                normalized_imgs = normalized_imgs.transpose(0, 2, 1, 3)

            nframes = normalized_imgs.shape[3]
            fios = AnnotationsApi._numpy_to_bytesio_png(normalized_imgs)

        elif file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            segs_imgs = nib.load(file_path).get_fdata()
            if segs_imgs.ndim != 3 and segs_imgs.ndim != 2:
                raise ValueError(f"Invalid segmentation shape: {segs_imgs.shape}")

            # Normalize and apply transpose
            normalized_imgs = AnnotationsApi._normalize_segmentation_array(segs_imgs)
            if not transpose_segmentation:
                # Apply default NIfTI transpose
                # (channels, width, height, frames) -> (channels, height, width, frames)
                normalized_imgs = normalized_imgs.transpose(0, 2, 1, 3)

            nframes = normalized_imgs.shape[3]
            fios = AnnotationsApi._numpy_to_bytesio_png(normalized_imgs)

        elif file_path.endswith('.png'):
            with Image.open(file_path) as img:
                img_array = np.array(img)
                normalized_imgs = AnnotationsApi._normalize_segmentation_array(img_array)

                if transpose_segmentation:
                    normalized_imgs = normalized_imgs.transpose(0, 2, 1, 3)

                fios = AnnotationsApi._numpy_to_bytesio_png(normalized_imgs)
                nframes = 1
        else:
            raise ValueError(f"Unsupported file format of '{file_path}'")

        return nframes, fios

    @staticmethod
    def _normalize_segmentation_array(seg_imgs: np.ndarray) -> np.ndarray:
        """
        Normalize segmentation array to a consistent format.

        Args:
            seg_imgs: Input segmentation array in various formats: (height, width, #frames), (height, width), (3, height, width, #frames).

        Returns:
            np.ndarray: Shape (#channels, height, width, #frames)
        """
        if seg_imgs.ndim == 4:
            return seg_imgs  # .transpose(1, 2, 0, 3)

        # Handle grayscale segmentations
        if seg_imgs.ndim == 2:
            # Add frame dimension: (height, width) -> (height, width, 1)
            seg_imgs = seg_imgs[..., None]
        if seg_imgs.ndim == 3:
            # (height, width, #frames)
            seg_imgs = seg_imgs[np.newaxis, ...]  # Add channel dimension: (1, height, width, #frames)

        return seg_imgs

    @staticmethod
    def _numpy_to_bytesio_png(seg_imgs: np.ndarray) -> Generator[BinaryIO, None, None]:
        """
        Convert normalized segmentation images to PNG BytesIO objects.

        Args:
            seg_imgs: Normalized segmentation array in shape (channels, height, width, frames).

        Yields:
            BinaryIO: PNG image data as BytesIO objects
        """
        # PIL RGB format is: (height, width, channels)
        if seg_imgs.shape[0] not in [1, 3, 4]:
            raise ValueError(f"Unsupported number of channels: {seg_imgs.shape[0]}. Expected 1 or 3")
        nframes = seg_imgs.shape[3]
        for i in range(nframes):
            img = seg_imgs[:, :, :, i].astype(np.uint8)
            if img.shape[0] == 1:
                pil_img = Image.fromarray(img[0]).convert('RGB')
            else:
                pil_img = Image.fromarray(img.transpose(1, 2, 0))
            img_bytes = BytesIO()
            pil_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            yield img_bytes

    def create_image_classification(self,
                                    resource: str | Resource,
                                    identifier: str,
                                    value: str,
                                    imported_from: str | None = None,
                                    model_id: str | None = None,
                                    ) -> str:
        """
        Create an image-level classification annotation.

        Args:
            resource: The resource unique id or Resource instance.
            identifier: The annotation identifier/label.
            value: The classification value.
            imported_from: The imported from source value.
            model_id: The model unique id.

        Returns:
            The id of the created annotation.
        """
        annotation_dto = CreateAnnotationDto(
            type=AnnotationType.CATEGORY,
            identifier=identifier,
            scope='image',
            value=value,
            imported_from=imported_from,
            model_id=model_id
        )

        return self.create(resource, annotation_dto)

    def add_line_annotation(self,
                            point1: tuple[int, int] | tuple[float, float, float],
                            point2: tuple[int, int] | tuple[float, float, float],
                            resource_id: str,
                            identifier: str,
                            frame_index: int | None = None,
                            dicom_metadata: pydicom.Dataset | str | None = None,
                            coords_system: CoordinateSystem = 'pixel',
                            project: str | None = None,
                            worklist_id: str | None = None,
                            imported_from: str | None = None,
                            author_email: str | None = None,
                            model_id: str | None = None) -> Sequence[str]:
        """
        Add a line annotation to a resource.

        Args:
            point1: The first point of the line. Can be a 2d or 3d point.
                If `coords_system` is 'pixel', it must be a 2d point and it represents the pixel coordinates of the image.
                If `coords_system` is 'patient', it must be a 3d point and it represents the patient coordinates of the image, relative
                to the DICOM metadata.
            If `coords_system` is 'patient', it must be a 3d point.
            point2: The second point of the line. See `point1` for more details.
            resource_id: The resource unique id.
            identifier: The annotation identifier, also as known as the annotation's label.
            frame_index: The frame index of the annotation.
            dicom_metadata: The DICOM metadata of the image. If provided, the coordinates will be converted to the
                correct coordinates automatically using the DICOM metadata.
            coords_system: The coordinate system of the points. Can be 'pixel', or 'patient'.
                If 'pixel', the points are in pixel coordinates. If 'patient', the points are in patient coordinates (see DICOM patient coordinates).
            project: The project unique id or name.
            worklist_id: The annotation worklist unique id. Optional.
            imported_from: The imported from source value.
            author_email: The email to consider as the author of the annotation. If None, use the customer of the api key.
            model_id: The model unique id. Optional.

        Example:
            .. code-block:: python

                res_id = 'aa93813c-cef0-4edd-a45c-85d4a8f1ad0d'
                api.add_line_annotation([0, 0], (10, 30),
                                        resource_id=res_id,
                                        identifier='Line1',
                                        frame_index=2,
                                        project='Example Project')
        """

        if project is not None and worklist_id is not None:
            raise ValueError('Only one of project or worklist_id can be provided.')

        if coords_system == 'pixel':
            if dicom_metadata is None:
                point1 = (point1[0], point1[1], frame_index)
                point2 = (point2[0], point2[1], frame_index)
                geom = LineGeometry(point1, point2)
            else:
                if isinstance(dicom_metadata, str):
                    dicom_metadata = pydicom.dcmread(dicom_metadata)
                geom = LineGeometry.from_dicom(dicom_metadata, point1, point2, slice_index=frame_index)
        elif coords_system == 'patient':
            geom = LineGeometry(point1, point2)
        else:
            raise ValueError(f"Unknown coordinate system: {coords_system}")

        return self._create_geometry_annotation(
            geometry=geom,
            resource_id=resource_id,
            identifier=identifier,
            frame_index=frame_index,
            project=project,
            worklist_id=worklist_id,
            imported_from=imported_from,
            author_email=author_email,
            model_id=model_id
        )

    def _create_geometry_annotation(self,
                                    geometry: Geometry,
                                    resource_id: str,
                                    identifier: str,
                                    frame_index: int | None = None,
                                    project: str | None = None,
                                    worklist_id: str | None = None,
                                    imported_from: str | None = None,
                                    author_email: str | None = None,
                                    model_id: str | None = None) -> Sequence[str]:
        """
        Create an annotation with the given geometry.

        Args:
            geometry: The geometry object (e.g., LineGeometry, BoxGeometry).
            resource_id: The resource unique id.
            identifier: The annotation identifier/label.
            frame_index: The frame index of the annotation.
            project: The project unique id or name.
            worklist_id: The annotation worklist unique id.
            imported_from: The imported from source value.
            author_email: The email to consider as the author.
            model_id: The model unique id.

        Returns:
            List of created annotation IDs.
        """
        if project is not None and worklist_id is not None:
            raise ValueError('Only one of project or worklist_id can be provided.')

        scope = 'frame' if frame_index is not None else 'image'
        annotation_dto = CreateAnnotationDto(
            type=geometry.type,
            identifier=identifier,
            scope=scope,
            frame_index=frame_index,
            geometry=geometry,
            imported_from=imported_from,
            import_author=author_email,
            model_id=model_id,
            annotation_worklist_id=worklist_id
        )

        return self.create(resource_id, annotation_dto)

    def download_file(self,
                      annotation: str | Annotation,
                      fpath_out: str | os.PathLike | None = None) -> bytes:
        """
        Download the segmentation file for a given resource and annotation.

        Args:
            annotation: The annotation unique id or an annotation object.
            fpath_out: (Optional) The file path to save the downloaded segmentation file.

        Returns:
            bytes: The content of the downloaded segmentation file in bytes format.
        """
        if isinstance(annotation, Annotation):
            annotation_id = annotation.id
            resource_id = annotation.resource_id
        else:
            annotation_id = annotation
            resource_id = self.get_by_id(annotation_id).resource_id

        resp = self._make_request('GET', f'/annotations/{resource_id}/annotations/{annotation_id}/file')
        if fpath_out:
            with open(fpath_out, 'wb') as f:
                f.write(resp.content)
        return resp.content

    async def _async_download_segmentation_file(self,
                                                annotation: str | Annotation,
                                                save_path: str | Path,
                                                session: aiohttp.ClientSession | None = None,
                                                progress_bar: tqdm | None = None):
        """
        Asynchronously download a segmentation file.

        Args:
            annotation (str | dict): The annotation unique id or an annotation object.
            save_path (str | Path): The path to save the file.
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            progress_bar (tqdm | None): Optional progress bar to update after download completion.

        Returns:
            dict: A dictionary with 'success' (bool) and optional 'error' (str) keys.
        """
        if isinstance(annotation, Annotation):
            annotation_id = annotation.id
            resource_id = annotation.resource_id
        else:
            annotation_id = annotation
            try:
                resource_id = self.get_by_id(annotation_id).resource_id
            except Exception as e:
                error_msg = f"Failed to get resource_id for annotation {annotation_id}: {str(e)}"
                _LOGGER.error(error_msg)
                if progress_bar:
                    progress_bar.update(1)
                return {'success': False, 'annotation_id': annotation_id, 'error': error_msg}

        try:
            async with self._make_request_async('GET',
                                                f'/annotations/{resource_id}/annotations/{annotation_id}/file',
                                                session=session) as resp:
                data_bytes = await resp.read()
                with open(save_path, 'wb') as f:
                    f.write(data_bytes)
            if progress_bar:
                progress_bar.update(1)
            return {'success': True, 'annotation_id': annotation_id}
        except Exception as e:
            error_msg = f"Failed to download annotation {annotation_id}: {str(e)}"
            _LOGGER.error(error_msg)
            if progress_bar:
                progress_bar.update(1)
            return {'success': False, 'annotation_id': annotation_id, 'error': error_msg}

    def download_multiple_files(self,
                                annotations: Sequence[str | Annotation],
                                save_paths: Sequence[str | Path] | str
                                ) -> list[dict[str, Any]]:
        """
        Download multiple segmentation files and save them to the specified paths.

        Args:
            annotations: A list of annotation unique ids or annotation objects.
            save_paths: A list of paths to save the files or a directory path.

        Returns:
            List of dictionaries with 'success', 'annotation_id', and optional 'error' keys.

        Note:
            If any downloads fail, they will be logged but the process will continue.
            A summary of failed downloads will be logged at the end.
        """
        import nest_asyncio
        nest_asyncio.apply()

        async def _download_all_async():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._async_download_segmentation_file(
                        annotation, save_path=path, session=session, progress_bar=progress_bar)
                    for annotation, path in zip(annotations, save_paths)
                ]
                return await asyncio.gather(*tasks)

        if isinstance(save_paths, str):
            save_paths = [os.path.join(save_paths, self._entid(ann))
                          for ann in annotations]

        with tqdm(total=len(annotations), desc="Downloading segmentations", unit="file") as progress_bar:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(_download_all_async())

        # Log summary of failures
        failures = [r for r in results if not r['success']]
        if failures:
            _LOGGER.warning(f"Failed to download {len(failures)} out of {len(annotations)} annotations")
            _USER_LOGGER.warning(f"Failed to download {len(failures)} out of {len(annotations)} annotations")
            for failure in failures:
                _LOGGER.debug(f"  - {failure['annotation_id']}: {failure['error']}")
        else:
            _USER_LOGGER.info(f"Successfully downloaded all {len(annotations)} annotations")

        return results

    def bulk_download_file(self,
                           annotations: Sequence[str | Annotation],
                           save_paths: Sequence[str | Path] | str
                           ) -> None:
        """Alias for :py:meth:`download_multiple_files`"""
        return self.download_multiple_files(annotations, save_paths)

    def patch(self,
              annotation: str | Annotation,
              identifier: str) -> None:
        """
        Update the project assignment for an annotation.

        Args:
            annotation: The annotation unique id or Annotation instance.
            identifier: The new identifier/label for the annotation.

        Raises:
            DatamintException: If the update fails.
        """
        annotation_id = self._entid(annotation)

        payload = {'identifier': identifier}
        # remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        if len(payload) == 0:
            _LOGGER.info("No fields to update for annotation patch.")
            return

        resp = self._make_request('PATCH',
                                  f'{self.endpoint_base}/{annotation_id}',
                                  json=payload)

        respdata = resp.json()
        if isinstance(respdata, dict) and 'error' in respdata:
            raise DatamintException(respdata['error'])

    def _get_resource(self, ann: Annotation) -> Resource:
        return self._resources_api.get_by_id(ann.resource_id)
