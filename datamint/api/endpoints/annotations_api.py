from typing import Any, Sequence, Literal, BinaryIO, Generator, IO
import httpx
from datetime import date
import logging
from ..base_api import EntityBaseApi, ApiConfig
from datamint.entities.annotation import Annotation
from datamint.entities.resource import Resource
from datamint.apihandler.dto.annotation_dto import AnnotationType, CreateAnnotationDto
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

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')
MAX_NUMBER_DISTINCT_COLORS = 2048  # Maximum number of distinct colors in a segmentation image


class AnnotationsApi(EntityBaseApi[Annotation]):
    """API handler for annotation-related endpoints."""

    def __init__(self, config: ApiConfig, client: httpx.Client | None = None) -> None:
        """Initialize the annotations API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        super().__init__(config, Annotation, 'annotations', client)

    # def create(self, annotation_data: dict[str, Any]) -> str:
    #     """Create a new annotation.

    #     Args:
    #         annotation_data: Dictionary payload for the annotation.

    #     Returns:
    #         The id of the created annotation.
    #     """
    #     return self._create(annotation_data)

    def get_list(self,
                 resource: str | Resource | None = None,
                 annotation_type: AnnotationType | str | None = None,
                 annotator_email: str | None = None,
                 date_from: date | None = None,
                 date_to: date | None = None,
                 dataset_id: str | None = None,
                 worklist_id: str | None = None,
                 status: Literal['new', 'published'] | None = None,
                 load_ai_segmentations: bool | None = None,
                 limit: int | None = None
                 ) -> Sequence[Annotation]:
        payload = {
            'resource_id': resource.id if isinstance(resource, Resource) else resource,
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
        return super().get_list(limit=limit, **payload)

    async def _upload_segmentations_async(self,
                                          resource_id: str,
                                          frame_index: int | None,
                                          file_path: str | np.ndarray,
                                          name: dict[int, str] | dict[tuple, str],
                                          imported_from: str | None = None,
                                          author_email: str | None = None,
                                          discard_empty_segmentations: bool = True,
                                          worklist_id: str | None = None,
                                          model_id: str | None = None,
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
        nframes, fios = AnnotationsApi._generate_segmentations_ios(
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
                                           resource_id: str,
                                           annotation_id: str,
                                           file: str | IO,
                                           content_type: str | None = None,
                                           filename: str | None = None
                                           ):
        """
        Upload a file for an existing annotation asynchronously.

        Args:
            resource_id: The resource unique id.
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
                    resource_id='your_resource_id',
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
            respdata = await self._make_request_async(method='POST',
                                                      endpoint=f'{self.endpoint_base}/{resource_id}/annotations/{annotation_id}/file',
                                                      data=form)
            if isinstance(respdata, dict) and 'error' in respdata:
                raise DatamintException(respdata['error'])
        finally:
            if close_file:
                f.close()

    def upload_annotation_file(self,
                               resource_id: str,
                               annotation_id: str,
                               file: str | IO,
                               content_type: str | None = None,
                               filename: str | None = None
                               ):
        """
        Upload a file for an existing annotation.

        Args:
            resource_id: The resource unique id.
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
        """Create a new annotation.

        Args:
            resource: The resource unique id or Resource instance.
            annotation_dto: A CreateAnnotationDto instance or a list of such instances.

        Returns:
            The id of the created annotation or a list of ids if multiple annotations were created.
        """

        annotations = [annotation_dto] if isinstance(annotation_dto, CreateAnnotationDto) else annotation_dto
        annotations = [ann.to_dict() if isinstance(ann, CreateAnnotationDto) else ann for ann in annotations]
        resource_id = resource.id if isinstance(resource, Resource) else resource
        respdata = self._make_request('POST',
                                      f'{self.endpoint_base}/{resource_id}/annotations',
                                      json=annotations).json()
        for r in respdata:
            if isinstance(r, dict) and 'error' in r:
                raise DatamintException(r['error'])
        if isinstance(annotation_dto, CreateAnnotationDto):
            return respdata[0]
        return respdata

    async def _create_async(self,
                            resource_id: str,
                            annotations_dto: list[CreateAnnotationDto] | list[dict]) -> list[str]:
        annotations = [ann.to_dict() if isinstance(ann, CreateAnnotationDto) else ann for ann in annotations_dto]
        respdata = await self._make_request_async('POST',
                                                  f'{self.endpoint_base}/{resource_id}/annotations',
                                                  data_to_get='json',
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
                    form.add_field('file', f, filename=filename, content_type=DEFAULT_NIFTI_MIME)
                    if model_id is not None:
                        form.add_field('model_id', model_id)  # Add model_id if provided
                    if worklist_id is not None:
                        form.add_field('annotation_worklist_id', worklist_id)
                    if name is not None:
                        form.add_field('segmentation_map', json.dumps(name), content_type='application/json')

                    resp = await self._make_request_async(method='POST',
                                                          endpoint=f'{self.endpoint_base}/{resource_id}/segmentations/file',
                                                          data=form,
                                                          data_to_get='json')
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
