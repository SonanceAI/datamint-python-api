from typing import Union, Tuple, Optional, Generator, Literal, List, Dict, IO
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
from deprecated.sphinx import deprecated

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')


class AnnotationAPIHandler(BaseAPIHandler):
    @staticmethod
    def _numpy_to_bytesio_png(seg_imgs: np.ndarray) -> Generator[BytesIO, None, None]:
        """
        Args:
            seg_img (np.ndarray): The segmentation image with dimensions (height, width, #frames).
        """

        if seg_imgs.ndim == 2:
            seg_imgs = seg_imgs[..., None]

        seg_imgs = seg_imgs.astype(np.uint8)
        for i in range(seg_imgs.shape[2]):
            img = seg_imgs[:, :, i]
            img = Image.fromarray(img).convert('L')
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            yield img_bytes

    @staticmethod
    def _generate_segmentations_ios(file_path: str | np.ndarray,
                                    transpose_segmentation: bool = False) -> tuple[int, Generator[IO, None, None]]:
        _LOGGER.debug(f'Generating segmentations io from file_path type: {type(file_path)}')
        if isinstance(file_path, np.ndarray):
            segs_imgs = file_path  # (#frames, height, width) or (height, width)
            if transpose_segmentation:
                segs_imgs = segs_imgs.transpose(1, 0, 2) if segs_imgs.ndim == 3 else segs_imgs.transpose(1, 0)
            nframes = segs_imgs.shape[2] if segs_imgs.ndim == 3 else 1
            fios = AnnotationAPIHandler._numpy_to_bytesio_png(segs_imgs)
        elif file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            segs_imgs = nib.load(file_path).get_fdata()
            if segs_imgs.ndim != 3 and segs_imgs.ndim != 2:
                raise ValueError(f"Invalid segmentation shape: {segs_imgs.shape}")
            if not transpose_segmentation:
                # The if is correct. The image is already in a different shape than nifty images.
                segs_imgs = segs_imgs.transpose(1, 0, 2) if segs_imgs.ndim == 3 else segs_imgs.transpose(1, 0)

            fios = AnnotationAPIHandler._numpy_to_bytesio_png(segs_imgs)
            nframes = segs_imgs.shape[2] if segs_imgs.ndim == 3 else 1
        elif file_path.endswith('.png'):
            if transpose_segmentation:
                with Image.open(file_path) as img:
                    segs_imgs = np.array(img).transpose(1, 0)
                fios = AnnotationAPIHandler._numpy_to_bytesio_png(segs_imgs)
            else:
                fios = (open(file_path, 'rb') for _ in range(1))
            nframes = 1
        else:
            raise ValueError(f"Unsupported file format of '{file_path}'")

        return nframes, fios

    async def _upload_annotations_async(self,
                                        resource_id: str,
                                        annotations: List[Dict]) -> List[str]:
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

    async def _upload_segmentations_async(self,
                                          resource_id: str,
                                          frame_index: int,
                                          file_path: str | None = None,
                                          fio: IO = None,
                                          name: Optional[Union[str, Dict[int, str]]] = None,
                                          imported_from: Optional[str] = None,
                                          author_email: Optional[str] = None,
                                          discard_empty_segmentations: bool = True,
                                          worklist_id: Optional[str] = None,
                                          model_id: Optional[str] = None,
                                          transpose_segmentation: bool = False
                                          ) -> None:
        if file_path is not None:
            nframes, fios = AnnotationAPIHandler._generate_segmentations_ios(file_path,
                                                                             transpose_segmentation=transpose_segmentation)
            if frame_index is None:
                frame_index = list(range(nframes))
            for fidx, f in zip(frame_index, fios):
                await self._upload_segmentations_async(resource_id,
                                                       fio=f,
                                                       name=name,
                                                       frame_index=fidx,
                                                       imported_from=imported_from,
                                                       author_email=author_email,
                                                       discard_empty_segmentations=discard_empty_segmentations,
                                                       worklist_id=worklist_id,
                                                       model_id=model_id)
            return
        try:
            try:
                img = np.array(Image.open(fio))
                ### Check that frame is not empty ###
                uniq_vals = np.unique(img)
                if discard_empty_segmentations:
                    if len(uniq_vals) == 1 and uniq_vals[0] == 0:
                        msg = f"Discarding empty segmentation for frame {frame_index}"
                        _LOGGER.debug(msg)
                        _USER_LOGGER.debug(msg)
                        return
                    fio.seek(0)
                    # TODO: Optimize this. It is not necessary to open the image twice.

                segnames = AnnotationAPIHandler._get_segmentation_names(uniq_vals, names=name)
                segs_generator = AnnotationAPIHandler._split_segmentations(img, uniq_vals, fio)
                annotations = []
                for segname in segnames:
                    ann = {
                        "identifier": segname,
                        "scope": 'frame',
                        "frame_index": frame_index,
                        'imported_from': imported_from,
                        'import_author': author_email,
                        "type": 'segmentation',
                        'annotation_worklist_id': worklist_id
                    }
                    if model_id is not None:
                        ann['model_id'] = model_id
                        ann['is_model'] = True
                    annotations.append(ann)
                # raise ValueError if there is multiple annotations with the same identifier, frame_index, scope and author
                if len(annotations) != len(set([a['identifier'] for a in annotations])):
                    raise ValueError(
                        "Multiple annotations with the same identifier, frame_index, scope and author is not supported yet.")

                annotids = await self._upload_annotations_async(resource_id, annotations)

                ### Upload segmentation ###
                if len(annotids) != len(segnames):
                    _LOGGER.warning(f"Number of uploaded annotations ({len(annotids)})" +
                                    f" does not match the number of annotations ({len(segnames)})")
                for annotid, segname, fio in zip(annotids, segnames, segs_generator):
                    form = aiohttp.FormData()
                    form.add_field('file', fio, filename=segname, content_type='image/png')
                    request_params = dict(
                        method='POST',
                        url=f'{self.root_url}/annotations/{resource_id}/annotations/{annotid}/file',
                        data=form,
                    )
                    resp = await self._run_request_async(request_params)
                    if 'error' in resp:
                        raise DatamintException(resp['error'])
                #######
            finally:
                fio.close()
            _USER_LOGGER.info(f'Segmentations uploaded for resource {resource_id}')
        except ResourceNotFoundError:
            raise ResourceNotFoundError('resource', {'resource_id': resource_id})

    def upload_segmentations(self,
                             resource_id: str,
                             file_path: Union[str, np.ndarray],
                             name: Optional[Union[str, Dict[int, str]]] = None,
                             frame_index: int | list[int] = None,
                             imported_from: Optional[str] = None,
                             author_email: Optional[str] = None,
                             discard_empty_segmentations: bool = True,
                             worklist_id: Optional[str] = None,
                             model_id: Optional[str] = None,
                             transpose_segmentation: bool = False
                             ) -> str:
        """
        Upload segmentations to a resource.

        Args:
            resource_id (str): The resource unique id.
            file_path (str|np.ndarray): The path to the segmentation file.
                If a numpy array is provided, it must have the shape (height, width, #frames) or (height, width).
            name (Optional[Union[str, Dict[int, str]]]): The name of the segmentation or a dictionary mapping pixel values to names.
                example: {1: 'Femur', 2: 'Tibia'}.
            frame_index (int | list[int]): The frame index of the segmentation. 
                If a list, it must have the same length as the number of frames in the segmentation.
                If None, it is assumed that the segmentations are in sequential order starting from 0.

            discard_empty_segmentations (bool): Whether to discard empty segmentations or not.

        Returns:
            str: The segmentation unique id.

        Raises:
            ResourceNotFoundError: If the resource does not exists or the segmentation is invalid.

        Example:
            >>> api_handler.upload_segmentation(resource_id, 'path/to/segmentation.png', 'SegmentationName')
        """
        if isinstance(file_path, str) and not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        if isinstance(frame_index, int):
            frame_index = [frame_index]

        loop = asyncio.get_event_loop()
        to_run = []
        # Generate IOs for the segmentations.
        nframes, fios = AnnotationAPIHandler._generate_segmentations_ios(file_path,
                                                                         transpose_segmentation=transpose_segmentation)
        if frame_index is None:
            frame_index = list(range(nframes))
        elif len(frame_index) != nframes:
            raise ValueError("Do not provide frame_index for images of multiple frames.")
        #######

        # For each frame, create the annotations and upload the segmentations.
        for fidx, f in zip(frame_index, fios):
            task = self._upload_segmentations_async(resource_id,
                                                    fio=f,
                                                    name=name,
                                                    frame_index=fidx,
                                                    imported_from=imported_from,
                                                    author_email=author_email,
                                                    discard_empty_segmentations=discard_empty_segmentations,
                                                    worklist_id=worklist_id,
                                                    model_id=model_id)
            to_run.append(task)

        ret = loop.run_until_complete(asyncio.gather(*to_run))
        return ret

    def add_image_category_annotation(self,
                                      resource_id: str,
                                      identifier: str,
                                      value: str,
                                      imported_from: Optional[str] = None,
                                      author_email: Optional[str] = None,
                                      worklist_id: Optional[str] = None
                                      ):

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
                                      frame_index: Union[int, Tuple[int, int]],
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
                        frame_index: Optional[Union[int, Tuple[int, int]]] = None,
                        value: Optional[str] = None,
                        worklist_id: Optional[str] = None,
                        imported_from: Optional[str] = None,
                        author_email: Optional[str] = None,
                        model_id: Optional[str] = None,
                        ):
        """
        Add annotations to a resource.

        Args:
            resource_id (str): The resource unique id.
            identifier (str): The annotation identifier.
            frame_index (Optional[Union[int, Tuple[int, int]]]): The frame index or a tuple with the range of frame indexes.
                If a tuple is provided, the annotation will be added to all frames in the range (Inclusive on both ends).
            value (Optional[str]): The annotation value.
            worklist_id (Optional[str]): The annotation worklist unique id.
            imported_from (Optional[str]): The imported from value.
            author_email (Optional[str]): The author email. If None, use the customer of the api key.
                Requires admin permissions to set a different customer.
            model_id (Optional[str]): The model unique id.
        """

        if isinstance(frame_index, tuple):
            frame_index = list(range(frame_index[0], frame_index[1]+1))
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
            'type': 'category',
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

    @deprecated(version='0.12.1', reason='Use :meth:`~get_annotations` instead with `resource_id` parameter.')
    def get_resource_annotations(self,
                                 resource_id: str,
                                 annotation_type: Optional[str] = None,
                                 annotator_email: Optional[str] = None,
                                 date_from: Optional[date] = None,
                                 date_to: Optional[date] = None) -> Generator[dict, None, None]:

        return self.get_annotations(resource_id=resource_id,
                                    annotation_type=annotation_type,
                                    annotator_email=annotator_email,
                                    date_from=date_from,
                                    date_to=date_to)

    def get_annotations(self,
                        resource_id: Optional[str] = None,
                        annotation_type: Optional[str] = None,
                        annotator_email: Optional[str] = None,
                        date_from: Optional[date] = None,
                        date_to: Optional[date] = None) -> Generator[dict, None, None]:
        """
        Get annotations for a resource.

        Args:
            resource_id (Optional[str]): The resource unique id.
            annotation_type (Optional[str]): The annotation type.
            annotator_email (Optional[str]): The annotator email.
            date_from (Optional[date]): The start date.
            date_to (Optional[date]): The end date.

        Returns:
            Generator[dict, None, None]: A generator of dictionaries with the annotations information.
        """
        # TODO: create annotation_type enum

        payload = {
            'resource_id': resource_id,
            'annotation_type': annotation_type,
            'annotatorEmail': annotator_email,
            'from': date_from.isoformat() if date_from is not None else None,
            'to': date_to.isoformat() if date_to is not None else None
        }

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
                                      id: str) -> Dict:
        """
        Get the annotation worklist.

        Args:
            id (str): The annotation worklist unique id.

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
                                   frame_labels: List[str] = None,
                                   image_labels: List[str] = None,
                                   annotations: List[Dict] = None,
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
    def _get_segmentation_names(uniq_vals: np.ndarray,
                                names: Optional[Union[str, Dict[int, str]]] = None
                                ) -> List[str]:
        uniq_vals = uniq_vals[uniq_vals != 0]
        if names is None:
            names = 'seg'
        if isinstance(names, str):
            if len(uniq_vals) == 1:
                return [names]
            return [f'{names}_{v}' for v in uniq_vals]
        if isinstance(names, dict):
            return [names.get(v, names.get('default', '')+'_'+str(v)) for v in uniq_vals]
        raise ValueError("names must be a string or a dictionary.")

    @staticmethod
    def _split_segmentations(img: np.ndarray,
                             uniq_vals: np.ndarray,
                             f: IO,
                             ) -> Generator[BytesIO, None, None]:
        # remove zero from uniq_vals
        uniq_vals = uniq_vals[uniq_vals != 0]

        for v in uniq_vals:
            img_v = (img == v).astype(np.uint8)

            f = BytesIO()
            Image.fromarray(img_v*255).convert('RGB').save(f, format='PNG')
            f.seek(0)
            yield f

    def delete_annotation(self, annotation_id: str):
        request_params = {
            'method': 'DELETE',
            'url': f'{self.root_url}/annotations/{annotation_id}',
        }

        resp = self._run_request(request_params)
        self._check_errors_response_json(resp)

    def get_segmentation_file(self, resource_id: str, annotation_id: str) -> bytes:
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/annotations/{resource_id}/annotations/{annotation_id}/file',
        }

        resp = self._run_request(request_params)
        return resp.content
