from typing import Optional, Generator, Literal, IO, BinaryIO
from .base_api_handler import BaseAPIHandler, ResourceNotFoundError, DatamintException
from datetime import date
import logging
import numpy as np
from PIL import Image
from io import BytesIO
import nibabel as nib
import os
import asyncio
import aiohttp
from requests.exceptions import HTTPError
from .dto.annotation_dto import CreateAnnotationDto, LineGeometry, BoxGeometry, CoordinateSystem, AnnotationType
import pydicom
import json
from deprecated import deprecated
from pathlib import Path
from tqdm.auto import tqdm

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')
MAX_NUMBER_DISTINCT_COLORS = 2048  # Maximum number of distinct colors in a segmentation image


class AnnotationAPIHandler(BaseAPIHandler):
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

    @staticmethod
    def _generate_segmentations_ios(file_path: str | np.ndarray,
                                    transpose_segmentation: bool = False) -> tuple[int, Generator[BinaryIO, None, None]]:
        if not isinstance(file_path, (str, np.ndarray)):
            raise ValueError(f"Unsupported file type: {type(file_path)}")

        if isinstance(file_path, np.ndarray):
            normalized_imgs = AnnotationAPIHandler._normalize_segmentation_array(file_path)
            # normalized_imgs shape: (3, height, width, #frames)

            # Apply transpose if requested
            if transpose_segmentation:
                # (channels, height, width, frames) -> (channels, width, height, frames)
                normalized_imgs = normalized_imgs.transpose(0, 2, 1, 3)

            nframes = normalized_imgs.shape[3]
            fios = AnnotationAPIHandler._numpy_to_bytesio_png(normalized_imgs)

        elif file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            segs_imgs = nib.load(file_path).get_fdata()
            if segs_imgs.ndim != 3 and segs_imgs.ndim != 2:
                raise ValueError(f"Invalid segmentation shape: {segs_imgs.shape}")

            # Normalize and apply transpose
            normalized_imgs = AnnotationAPIHandler._normalize_segmentation_array(segs_imgs)
            if not transpose_segmentation:
                # Apply default NIfTI transpose
                # (channels, width, height, frames) -> (channels, height, width, frames)
                normalized_imgs = normalized_imgs.transpose(0, 2, 1, 3)

            nframes = normalized_imgs.shape[3]
            fios = AnnotationAPIHandler._numpy_to_bytesio_png(normalized_imgs)

        elif file_path.endswith('.png'):
            with Image.open(file_path) as img:
                img_array = np.array(img)
                normalized_imgs = AnnotationAPIHandler._normalize_segmentation_array(img_array)

                if transpose_segmentation:
                    normalized_imgs = normalized_imgs.transpose(0, 2, 1, 3)

                fios = AnnotationAPIHandler._numpy_to_bytesio_png(normalized_imgs)
                nframes = 1
        else:
            raise ValueError(f"Unsupported file format of '{file_path}'")

        return nframes, fios

    async def _upload_annotations_async(self,
                                        resource_id: str,
                                        annotations: list[dict | CreateAnnotationDto]) -> list[str]:
        annotations = [ann.to_dict() if isinstance(ann, CreateAnnotationDto) else ann for ann in annotations]
        request_params = dict(
            method='POST',
            url=f'{self.root_url}/annotations/{resource_id}/annotations',
            json=annotations
        )
        resp = await self._run_request_async(request_params)
        for r in resp:
            if 'error' in r:
                raise DatamintException(r['error'])
        return resp

    async def _upload_single_frame_segmentation_async(self,
                                                      resource_id: str,
                                                      frame_index: int | None,
                                                      fio: IO,
                                                      name: dict[int, str] | dict[tuple, str],
                                                      imported_from: Optional[str] = None,
                                                      author_email: Optional[str] = None,
                                                      discard_empty_segmentations: bool = True,
                                                      worklist_id: Optional[str] = None,
                                                      model_id: Optional[str] = None
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
                segnames = AnnotationAPIHandler._get_segmentation_names_rgb(unique_vals, names=name)
                segs_generator = AnnotationAPIHandler._split_rgb_segmentations(img_array, unique_vals)

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

                annotids = await self._upload_annotations_async(resource_id, annotations)

                # Upload segmentation files
                if len(annotids) != len(segnames):
                    _LOGGER.warning(f"Number of uploaded annotations ({len(annotids)})" +
                                    f" does not match the number of annotations ({len(segnames)})")

                for annotid, segname, fio_seg in zip(annotids, segnames, segs_generator):
                    form = aiohttp.FormData()
                    form.add_field('file', fio_seg, filename=segname, content_type='image/png')
                    request_params = dict(
                        method='POST',
                        url=f'{self.root_url}/annotations/{resource_id}/annotations/{annotid}/file',
                        data=form,
                    )
                    resp = await self._run_request_async(request_params)
                    if 'error' in resp:
                        raise DatamintException(resp['error'])
                return annotids
            finally:
                fio.close()
        except ResourceNotFoundError:
            raise ResourceNotFoundError('resource', {'resource_id': resource_id})

    async def _upload_volume_segmentation_async(self,
                                                resource_id: str,
                                                file_path: str | np.ndarray,
                                                name: str | dict[int, str] | dict[tuple, str] | None,
                                                imported_from: Optional[str] = None,
                                                author_email: Optional[str] = None,
                                                worklist_id: Optional[str] = None,
                                                model_id: Optional[str] = None,
                                                transpose_segmentation: bool = False
                                                ) -> list[str]:
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

        # Prepare file for upload
        if isinstance(file_path, str):
            if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
                # Upload NIfTI file directly
                with open(file_path, 'rb') as f:
                    filename = os.path.basename(file_path)
                    form = aiohttp.FormData()
                    form.add_field('file', f, filename=filename, content_type='application/x-nifti')
                    if model_id is not None:
                        form.add_field('model_id', model_id)  # Add model_id if provided
                    if worklist_id is not None:
                        form.add_field('annotation_worklist_id', worklist_id)
                    if name is not None:
                        form.add_field('segmentation_map', json.dumps(name), content_type='application/json')

                    request_params = dict(
                        method='POST',
                        url=f'{self.root_url}/annotations/{resource_id}/segmentations/file',
                        data=form,
                    )
                    resp = await self._run_request_async(request_params)
                    if 'error' in resp:
                        raise DatamintException(resp['error'])
                    return resp
            else:
                raise ValueError(f"Volume upload not supported for file format: {file_path}")
        elif isinstance(file_path, np.ndarray):
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported file_path type for volume upload: {type(file_path)}")

        _USER_LOGGER.info(f'Volume segmentation uploaded for resource {resource_id}')

    async def _upload_segmentations_async(self,
                                          resource_id: str,
                                          frame_index: int | None,
                                          file_path: str | np.ndarray,
                                          name: dict[int, str] | dict[tuple, str],
                                          imported_from: Optional[str] = None,
                                          author_email: Optional[str] = None,
                                          discard_empty_segmentations: bool = True,
                                          worklist_id: Optional[str] = None,
                                          model_id: Optional[str] = None,
                                          transpose_segmentation: bool = False,
                                          upload_volume: bool | str = 'auto'
                                          ) -> list[str]:
        """
        Upload segmentations asynchronously.

        Args:
            resource_id: The resource unique id.
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
        nframes, fios = AnnotationAPIHandler._generate_segmentations_ios(
            file_path, transpose_segmentation=transpose_segmentation
        )
        if frame_index is None:
            frames_indices = list(range(nframes))
        elif isinstance(frame_index, int):
            frames_indices = [frame_index]
        else:
            raise ValueError("frame_index must be an int or None")

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

    @staticmethod
    def standardize_segmentation_names(name: str | dict[int, str] | dict[tuple, str] | None) -> dict[tuple[int, int, int], str]:
        if name is None:
            return {}
        elif isinstance(name, str):
            return {'default': name}
        elif isinstance(name, dict):
            name = {
                tuple(k) if isinstance(k, (list, tuple)) else k if isinstance(k, str) else (k, k, k): v
                for k, v in name.items()
            }
            if 'default' not in name:
                name['default'] = None
            return name
        else:
            raise ValueError("Invalid name format")

    def upload_segmentations(self,
                             resource_id: str,
                             file_path: str | np.ndarray,
                             name: str | dict[int, str] | dict[tuple, str] | None = None,
                             frame_index: int | list[int] | None = None,
                             imported_from: Optional[str] = None,
                             author_email: Optional[str] = None,
                             discard_empty_segmentations: bool = True,
                             worklist_id: Optional[str] = None,
                             model_id: Optional[str] = None,
                             transpose_segmentation: bool = False
                             ) -> list[str]:
        """
        Upload segmentations to a resource.

        Args:
            resource_id (str): The resource unique id.
            file_path (str|np.ndarray): The path to the segmentation file or a numpy array.
                If a numpy array is provided, it can have the shape:
                - (height, width, #frames) or (height, width) for grayscale segmentations
                - (3, height, width, #frames) for RGB segmentations
                For NIfTI files (.nii/.nii.gz), the entire volume is uploaded as a single segmentation.
            name: The name of the segmentation.
                Can be:
                - str: Single name for all segmentations
                - dict[int, str]: Mapping pixel values to names for grayscale segmentations
                - dict[tuple[int, int, int], str]: Mapping RGB tuples to names for RGB segmentations
                Example: {(255, 0, 0): 'Red_Region', (0, 255, 0): 'Green_Region'}
            frame_index (int | list[int]): The frame index of the segmentation. 
                If a list, it must have the same length as the number of frames in the segmentation.
                If None, it is assumed that the segmentations are in sequential order starting from 0.
                This parameter is ignored for NIfTI files as they are treated as volume segmentations.
            discard_empty_segmentations (bool): Whether to discard empty segmentations or not.

        Returns:
            List[str]: List of segmentation unique ids.

        Raises:
            ResourceNotFoundError: If the resource does not exists or the segmentation is invalid.

        Example:
            >>> # Grayscale segmentation
            >>> api_handler.upload_segmentation(resource_id, 'path/to/segmentation.png', 'SegmentationName')
            >>> 
            >>> # RGB segmentation with numpy array
            >>> seg_data = np.random.randint(0, 3, size=(3, 2140, 1760, 1), dtype=np.uint8)
            >>> rgb_names = {(1, 0, 0): 'Red_Region', (0, 1, 0): 'Green_Region', (0, 0, 1): 'Blue_Region'}
            >>> api_handler.upload_segmentation(resource_id, seg_data, rgb_names)
            >>>
            >>> # Volume segmentation
            >>> api_handler.upload_segmentation(resource_id, 'path/to/segmentation.nii.gz', 'VolumeSegmentation')
        """

        if isinstance(file_path, str) and not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        # Handle NIfTI files specially - upload as single volume
        if isinstance(file_path, str) and (file_path.endswith('.nii') or file_path.endswith('.nii.gz')):
            _LOGGER.info(f"Uploading NIfTI segmentation file: {file_path}")
            if frame_index is not None:
                raise ValueError("Do not provide frame_index for NIfTI segmentations.")
            loop = asyncio.get_event_loop()
            task = self._upload_volume_segmentation_async(
                resource_id=resource_id,
                file_path=file_path,
                name=name,
                imported_from=imported_from,
                author_email=author_email,
                worklist_id=worklist_id,
                model_id=model_id,
                transpose_segmentation=transpose_segmentation
            )
            return loop.run_until_complete(task)
        # All other file types are converted to multiple PNGs and uploaded frame by frame.

        name = AnnotationAPIHandler.standardize_segmentation_names(name)

        to_run = []
        # Generate IOs for the segmentations.
        nframes, fios = AnnotationAPIHandler._generate_segmentations_ios(file_path,
                                                                         transpose_segmentation=transpose_segmentation)
        if frame_index is None:
            frame_index = list(range(nframes))
        elif isinstance(frame_index, int):
            frame_index = [frame_index]
        if len(frame_index) != nframes:
            raise ValueError(f'Expected {nframes} frame_index values, but got {len(frame_index)}.')

        # For each frame, create the annotations and upload the segmentations.
        for fidx, f in zip(frame_index, fios):
            task = self._upload_single_frame_segmentation_async(resource_id,
                                                                fio=f,
                                                                name=name,
                                                                frame_index=fidx,
                                                                imported_from=imported_from,
                                                                author_email=author_email,
                                                                discard_empty_segmentations=discard_empty_segmentations,
                                                                worklist_id=worklist_id,
                                                                model_id=model_id)
            to_run.append(task)

        loop = asyncio.get_event_loop()
        ret = loop.run_until_complete(asyncio.gather(*to_run))
        # merge the results in a single list
        ret = [item for sublist in ret for item in sublist]
        return ret

    def add_image_category_annotation(self,
                                      resource_id: str,
                                      identifier: str,
                                      value: str,
                                      imported_from: Optional[str] = None,
                                      author_email: Optional[str] = None,
                                      worklist_id: Optional[str] = None,
                                      project: Optional[str] = None
                                      ):
        """
        Add a category annotation to an image.

        Args:
            resource_id (str): The resource unique id.
            identifier (str): The annotation identifier. For example: 'fracture'.
            value (str): The annotation value. 
            imported_from (Optional[str]): The imported from value.
            author_email (Optional[str]): The author email. If None, use the customer of the api key.
            wokklist_id (Optional[str]): The annotation worklist unique id.
            project (Optional[str]): The project unique id or name. Only this or worklist_id can be provided at the same time.

        """
        if project is not None and worklist_id is not None:
            raise ValueError('Only one of project or worklist_id can be provided.')
        if project is not None:
            proj = self.get_project_by_name(project)
            if 'error' in proj.keys():
                raise DatamintException(f"Project {project} not found.")
            worklist_id = proj['worklist_id']

        if value is None:
            raise ValueError('Value cannot be None.')

        request_params = {
            'method': 'POST',
            'url': f'{self.root_url}/annotations/{resource_id}/annotations',
            'json': [{
                'identifier': identifier,
                'value': value,
                'scope': 'image',
                'type': 'category',
                'imported_from': imported_from,
                'import_author': author_email,
                'annotation_worklist_id': worklist_id
            }]
        }

        resp = self._run_request(request_params)
        self._check_errors_response_json(resp)

    def add_frame_category_annotation(self,
                                      resource_id: str,
                                      frame_index: int | tuple[int, int],
                                      identifier: str,
                                      value: str,
                                      worklist_id: Optional[str] = None,
                                      imported_from: Optional[str] = None,
                                      author_email: Optional[str] = None
                                      ):
        """
        Add a category annotation to a frame.

        Args:
            resource_id (str): The resource unique id.
            frame_index (Union[int, Tuple[int, int]]): The frame index or a tuple with the range of frame indexes.
                If a tuple is provided, the annotation will be added to all frames in the range (Inclusive on both ends).
            identifier (str): The annotation identifier.
            value (str): The annotation value.
            worklist_id (Optional[str]): The annotation worklist unique id.
            author_email (Optional[str]): The author email. If None, use the customer of the api key.
                Requires admin permissions to set a different customer.
        """

        if isinstance(frame_index, tuple):
            frame_index = list(range(frame_index[0], frame_index[1]+1))
        elif isinstance(frame_index, int):
            frame_index = [frame_index]

        json_data = [{
            'identifier': identifier,
            'value': value,
            'scope': 'frame',
            'frame_index': i,
            'annotation_worklist_id': worklist_id,
            'imported_from': imported_from,
            'import_author': author_email,
            'type': 'category'} for i in frame_index]

        request_params = {
            'method': 'POST',
            'url': f'{self.root_url}/annotations/{resource_id}/annotations',
            'json': json_data
        }

        resp = self._run_request(request_params)
        self._check_errors_response_json(resp)

    def add_annotations(self,
                        resource_id: str,
                        identifier: str,
                        frame_index: int | tuple[int, int] | None = None,
                        value: Optional[str] = None,
                        worklist_id: Optional[str] = None,
                        imported_from: Optional[str] = None,
                        author_email: Optional[str] = None,
                        model_id: Optional[str] = None,
                        project: Optional[str] = None,
                        ) -> list[str]:
        """
        Add annotations to a resource.

        Args:
            resource_id: The resource unique id.
            identifier: The annotation identifier.
            frame_index: The frame index or a tuple with the range of frame indexes.
                If a tuple is provided, the annotation will be added to all frames in the range (Inclusive on both ends).
            value: The annotation value.
            worklist_id: The annotation worklist unique id.
            imported_from: The imported from value.
            author_email: The author email. If None, use the customer of the api key.
                Requires admin permissions to set a different customer.
            model_id: The model unique id.
            project: The project unique id or name. Only this or worklist_id can be provided at the same time.
        """

        if project is not None and worklist_id is not None:
            raise ValueError('Only one of project or worklist_id can be provided.')
        if project is not None:
            proj = self.get_project_by_name(project)
            if 'error' in proj.keys():
                raise DatamintException(f"Project {project} not found.")
            worklist_id = proj['worklist_id']

        if isinstance(frame_index, tuple):
            begin, end = frame_index
            if begin > end:
                raise ValueError('The first element of the tuple must be less than the second element.')
            frame_index = list(range(begin, end+1))
        elif isinstance(frame_index, int):
            frame_index = [frame_index]

        scope = 'frame' if frame_index is not None else 'image'

        params = {
            'identifier': identifier,
            'value': value,
            'scope': scope,
            'annotation_worklist_id': worklist_id,
            'imported_from': imported_from,
            'import_author': author_email,
            'type': 'label' if value is None else 'category',
        }
        if model_id is not None:
            params['model_id'] = model_id
            params['is_model'] = True

        if frame_index is not None:
            json_data = [dict(params, frame_index=i) for i in frame_index]
        else:
            json_data = [params]

        request_params = {
            'method': 'POST',
            'url': f'{self.root_url}/annotations/{resource_id}/annotations',
            'json': json_data
        }

        resp = self._run_request(request_params)
        self._check_errors_response_json(resp)
        return resp.json()

    def _create_geometry_annotation(self,
                                    geometry: LineGeometry | BoxGeometry,
                                    resource_id: str,
                                    identifier: str,
                                    frame_index: int | None = None,
                                    project: Optional[str] = None,
                                    worklist_id: Optional[str] = None,
                                    imported_from: Optional[str] = None,
                                    author_email: Optional[str] = None,
                                    model_id: Optional[str] = None) -> list[str]:
        """
        Common method for creating geometry-based annotations.

        Args:
            geometry: The geometry object (LineGeometry or BoxGeometry)
            resource_id: The resource unique id
            identifier: The annotation identifier
            frame_index: The frame index of the annotation
            project: The project unique id or name
            worklist_id: The annotation worklist unique id
            imported_from: The imported from source value
            author_email: The email to consider as the author of the annotation
            model_id: The model unique id
        """
        if project is not None and worklist_id is not None:
            raise ValueError('Only one of project or worklist_id can be provided.')

        if project is not None:
            proj = self.get_project_by_name(project)
            if 'error' in proj.keys():
                raise DatamintException(f"Project {project} not found.")
            worklist_id = proj['worklist_id']

        anndto = CreateAnnotationDto(
            type=geometry.type,
            identifier=identifier,
            scope='frame',
            annotation_worklist_id=worklist_id,
            value=None,
            imported_from=imported_from,
            import_author=author_email,
            frame_index=frame_index,
            geometry=geometry,
            model_id=model_id,
            is_model=model_id is not None,
        )

        json_data = anndto.to_dict()

        request_params = {
            'method': 'POST',
            'url': f'{self.root_url}/annotations/{resource_id}/annotations',
            'json': [json_data]
        }

        resp = self._run_request(request_params)
        self._check_errors_response_json(resp)
        return resp.json()

    def add_line_annotation(self,
                            point1: tuple[int, int] | tuple[float, float, float],
                            point2: tuple[int, int] | tuple[float, float, float],
                            resource_id: str,
                            identifier: str,
                            frame_index: int | None = None,
                            dicom_metadata: pydicom.Dataset | str | None = None,
                            coords_system: CoordinateSystem = 'pixel',
                            project: Optional[str] = None,
                            worklist_id: Optional[str] = None,
                            imported_from: Optional[str] = None,
                            author_email: Optional[str] = None,
                            model_id: Optional[str] = None) -> list[str]:
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

    def add_box_annotation(self,
                           point1: tuple[int, int] | tuple[float, float, float],
                           point2: tuple[int, int] | tuple[float, float, float],
                           resource_id: str,
                           identifier: str,
                           frame_index: int | None = None,
                           dicom_metadata: pydicom.Dataset | str | None = None,
                           coords_system: CoordinateSystem = 'pixel',
                           project: Optional[str] = None,
                           worklist_id: Optional[str] = None,
                           imported_from: Optional[str] = None,
                           author_email: Optional[str] = None,
                           model_id: Optional[str] = None):
        """
        Add a box annotation to a resource.

        Args:
            point1: The first corner point of the box. Can be a 2d or 3d point.
                If `coords_system` is 'pixel', it must be a 2d point representing pixel coordinates.
                If `coords_system` is 'patient', it must be a 3d point representing patient coordinates.
            point2: The opposite diagonal corner point of the box. See `point1` for more details.
            resource_id: The resource unique id.
            identifier: The annotation identifier, also known as the annotation's label.
            frame_index: The frame index of the annotation.
            dicom_metadata: The DICOM metadata of the image. If provided, coordinates will be converted
                automatically using the DICOM metadata.
            coords_system: The coordinate system of the points. Can be 'pixel' or 'patient'.
                If 'pixel', points are in pixel coordinates. If 'patient', points are in patient coordinates.
            project: The project unique id or name.
            worklist_id: The annotation worklist unique id. Optional.
            imported_from: The imported from source value.
            author_email: The email to consider as the author of the annotation. If None, uses the API key customer.
            model_id: The model unique id. Optional.

        Example:
            .. code-block:: python

                res_id = 'aa93813c-cef0-4edd-a45c-85d4a8f1ad0d'
                api.add_box_annotation([10, 10], (50, 40),
                                       resource_id=res_id,
                                       identifier='BoundingBox1',
                                       frame_index=2,
                                       project='Example Project')
        """
        if coords_system == 'pixel':
            if dicom_metadata is None:
                point1 = (point1[0], point1[1], frame_index)
                point2 = (point2[0], point2[1], frame_index)
                geom = BoxGeometry(point1, point2)
            else:
                if isinstance(dicom_metadata, str):
                    dicom_metadata = pydicom.dcmread(dicom_metadata)
                geom = BoxGeometry.from_dicom(dicom_metadata, point1, point2, slice_index=frame_index)
        elif coords_system == 'patient':
            geom = BoxGeometry(point1, point2)
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

    def get_annotations(self,
                        resource_id: Optional[str] = None,
                        annotation_type: AnnotationType | str | None = None,
                        annotator_email: Optional[str] = None,
                        date_from: Optional[date] = None,
                        date_to: Optional[date] = None,
                        dataset_id: Optional[str] = None,
                        worklist_id: Optional[str] = None,
                        status: Optional[Literal['new', 'published']] = None,
                        load_ai_segmentations: bool | None = None,
                        ) -> Generator[dict, None, None]:
        """
        Get annotations for a resource.

        Args:
            resource_id (Optional[str]): The resource unique id.
            annotation_type (AnnotationType | str | None): The annotation type. See :class:`~datamint.dto.annotation_dto.AnnotationType`.
            annotator_email (Optional[str]): The annotator email.
            date_from (Optional[date]): The start date.
            date_to (Optional[date]): The end date.
            dataset_id (Optional[str]): The dataset unique id.
            worklist_id (Optional[str]): The annotation worklist unique id.
            status (Optional[Literal['new', 'published']]): The status of the annotations.
            load_ai_segmentations (bool): Whether to load the AI segmentations or not. Default is False.

        Returns:
            Generator[dict, None, None]: A generator of dictionaries with the annotations information.
        """

        if annotation_type is not None and isinstance(annotation_type, AnnotationType):
            annotation_type = annotation_type.value

        payload = {
            'resource_id': resource_id,
            'annotation_type': annotation_type,
            'annotatorEmail': annotator_email,
            'from': date_from.isoformat() if date_from is not None else None,
            'to': date_to.isoformat() if date_to is not None else None,
            'dataset_id': dataset_id,
            'annotation_worklist_id': worklist_id,
            'status': status,
            'load_ai_segmentations': load_ai_segmentations
        }

        # remove nones
        payload = {k: v for k, v in payload.items() if v is not None}

        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/annotations',
            'params': payload
        }

        yield from self._run_pagination_request(request_params, return_field='data')

    def get_annotation_worklist(self,
                                status: Literal['new', 'updating', 'active', 'completed'] = None
                                ) -> Generator[dict, None, None]:
        """
        Get the annotation worklist.

        Args:
            status (Literal['new', 'updating','active', 'completed']): The status of the annotations.

        Returns:
            Generator[dict, None, None]: A generator of dictionaries with the annotations information.
        """

        payload = {}

        if status is not None:
            payload['status'] = status

        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/annotationsets',
            'params': payload
        }

        yield from self._run_pagination_request(request_params, return_field='data')

    def get_annotation_worklist_by_id(self,
                                      id: str) -> dict:
        """Get the annotation worklist.

        Args:
            id: The annotation worklist unique id.

        Returns:
            Dict: A dictionary with the annotations information.
        """

        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/annotationsets/{id}',
        }

        try:
            resp = self._run_request(request_params).json()
            return resp
        except HTTPError as e:
            if e.response.status_code == 404:
                raise ResourceNotFoundError('annotation worklist', {'id': id})
            raise e

    def update_annotation_worklist(self,
                                   worklist_id: str,
                                   frame_labels: list[str] = None,
                                   image_labels: list[str] = None,
                                   annotations: list[dict] = None,
                                   status: Literal['new', 'updating', 'active', 'completed'] = None,
                                   name: str = None,
                                   ):
        """
        Update the status of an annotation worklist.

        Args:
            worklist_id (str): The annotation worklist unique id.
            frame_labels (List[str]): The frame labels.
            image_labels (List[str]): The image labels.
            annotations (List[Dict]): The annotations.
            status (Literal['new', 'updating','active', 'completed']): The status of the annotations.

        """

        payload = {}
        if status is not None:
            payload['status'] = status
        if frame_labels is not None:
            payload['frame_labels'] = frame_labels
        if image_labels is not None:
            payload['image_labels'] = image_labels
        if annotations is not None:
            payload['annotations'] = annotations
        if name is not None:
            payload['name'] = name

        request_params = {
            'method': 'PATCH',
            'url': f'{self.root_url}/annotationsets/{worklist_id}',
            'json': payload
        }

        self._run_request(request_params)

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

    def delete_annotation(self, annotation_id: str | dict):
        if isinstance(annotation_id, dict):
            annotation_id = annotation_id.get('id', None)
            if annotation_id is None:
                raise ValueError("annotation_id must be a string or a dict with 'id' key.")
        request_params = {
            'method': 'DELETE',
            'url': f'{self.root_url}/annotations/{annotation_id}',
        }

        resp = self._run_request(request_params)
        self._check_errors_response_json(resp)

    def get_annotation_by_id(self, annotation_id: str) -> dict:
        """
        Get an annotation by its unique id.

        Args:
            annotation_id (str): The annotation unique id.

        Returns:
            dict: The annotation information.
        """
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/annotations/{annotation_id}',
        }

        try:
            resp = self._run_request(request_params)
            return resp.json()
        except HTTPError as e:
            _LOGGER.error(f"Error getting annotation by id {annotation_id}: {e}")
            raise

    @deprecated(reason="Use download_segmentation_file instead")
    def get_segmentation_file(self, resource_id: str, annotation_id: str) -> bytes:
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/annotations/{resource_id}/annotations/{annotation_id}/file',
        }

        resp = self._run_request(request_params)
        return resp.content

    def download_segmentation_file(self, annotation: str | dict, fpath_out: str | Path | None) -> bytes:
        """
        Download the segmentation file for a given resource and annotation.

        Args:
            annotation (str | dict): The annotation unique id or an annotation object.
            fpath_out (str | None): (Optional) The file path to save the downloaded segmentation file.

        Returns:
            bytes: The content of the downloaded segmentation file in bytes format.
        """
        if isinstance(annotation, dict):
            annotation_id = annotation['id']
            resource_id = annotation['resource_id']
        else:
            annotation_id = annotation
            resource_id = self.get_annotation_by_id(annotation_id)['resource_id']

        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/annotations/{resource_id}/annotations/{annotation_id}/file',
        }

        resp = self._run_request(request_params)
        if fpath_out is not None:
            with open(str(fpath_out), 'wb') as f:
                f.write(resp.content)
        return resp.content

    def set_annotation_status(self,
                              project_id: str,
                              resource_id: str,
                              status: Literal['opened', 'annotated', 'closed']
                              ):

        if status not in ['opened', 'annotated', 'closed']:
            raise ValueError("status must be one of ['opened', 'annotated', 'closed']")
        request_params = {
            'method': 'POST',
            'url': f'{self.root_url}/projects/{project_id}/resources/{resource_id}/status',
            'json': {
                'status': status
            }
        }
        resp = self._run_request(request_params)
        self._check_errors_response_json(resp)


    async def _async_download_segmentation_file(self,
                                                annotation: str | dict,
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
        """
        if isinstance(annotation, dict):
            annotation_id = annotation['id']
            resource_id = annotation['resource_id']
        else:
            annotation_id = annotation
            # TODO: This is inefficient as it requires an extra API call per annotation
            # Consider passing resource_id separately or caching annotation info
            resource_id = self.get_annotation_by_id(annotation_id)['resource_id']

        url = f'{self.root_url}/annotations/{resource_id}/annotations/{annotation_id}/file'
        request_params = {
            'method': 'GET',
            'url': url
        }
        
        try:
            data_bytes = await self._run_request_async(request_params, session, 'content')
            with open(save_path, 'wb') as f:
                f.write(data_bytes)
            if progress_bar:
                progress_bar.update(1)
        except ResourceNotFoundError as e:
            e.set_params('annotation', {'annotation_id': annotation_id})
            raise e

    def download_multiple_segmentations(self,
                                        annotations: list[str | dict],
                                        save_paths: list[str | Path] | str
                                        ) -> None:
        """
        Download multiple segmentation files and save them to the specified paths.
        
        Args:
            annotations (list[str | dict]): A list of annotation unique ids or annotation objects.
            save_paths (list[str | Path] | str): A list of paths to save the files or a directory path.
        """
        async def _download_all_async():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._async_download_segmentation_file(annotation, save_path=path, session=session, progress_bar=progress_bar)
                    for annotation, path in zip(annotations, save_paths)
                ]
                await asyncio.gather(*tasks)

        if isinstance(save_paths, str):
            save_paths = [os.path.join(save_paths, f"{ann['id'] if isinstance(ann, dict) else ann}") for ann in annotations]

        with tqdm(total=len(annotations), desc="Downloading segmentations", unit="file") as progress_bar:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_download_all_async())
