from .base_dataset import DatamintBaseDataset
from typing import List, Optional, Callable, Any, Dict, Literal
import torch
from torch import Tensor
import os
import numpy as np
import logging
from PIL import Image
import albumentations
from datamint.dataset.annotation import Annotation

_LOGGER = logging.getLogger(__name__)


class DatamintDataset(DatamintBaseDataset):
    """
    This Dataset class extends the `DatamintBaseDataset` class to be easily used with PyTorch.
    In addition to that, it has functionality to better process annotations and segmentations.

    .. note:: 
        Import using ``from datamint import Dataset``.

    Args:
        root: Root directory of dataset where data already exists or will be downloaded.
        project_name: Name of the project to download.
        auto_update: If True, the dataset will be checked for updates and downloaded if necessary.
        api_key: API key to access the Datamint API. If not provided, it will look for the
            environment variable 'DATAMINT_API_KEY'. Not necessary if
            you don't want to download/update the dataset.
        return_dicom: If True, the DICOM object will be returned, if the image is a DICOM file.
        return_metainfo: If True, the metainfo of the image will be returned.
        return_annotations: If True, the annotations of the image will be returned.
        return_frame_by_frame: If True, each frame of a video/DICOM/3d-image will be returned separately.
        include_unannotated: If True, images without annotations will be included. If False, images without annotations will be discarded. 
        all_annotations: If True, all annotations will be downloaded, including the ones that are not set as closed/done.
        server_url: URL of the Datamint server. If not provided, it will use the default server.
        return_segmentations: If True (default), the segmentations of the image will be returned in the 'segmentations' key.
        return_as_semantic_segmentation: If True, the segmentations will be returned as semantic segmentation.
        image_transform: A function to transform the image.
        mask_transform: A function to transform the mask.
        semantic_seg_merge_strategy: If not None, the segmentations will be merged using this strategy.
            Possible values are 'union', 'intersection', 'mode'.
        include_annotators: List of annotators to include. If None, all annotators will be included. See parameter ``exclude_annotators``.
        exclude_annotators: List of annotators to exclude. If None, no annotators will be excluded. See parameter ``include_annotators``.
        include_segmentation_names: List of segmentation names to include. If None, all segmentations will be included.
        exclude_segmentation_names: List of segmentation names to exclude. If None, no segmentations will be excluded.
        include_image_label_names: List of image label names to include. If None, all image labels will be included.
        exclude_image_label_names: List of image label names to exclude. If None, no image labels will be excluded.
        include_frame_label_names: List of frame label names to include. If None, all frame labels will be included.
        exclude_frame_label_names: List of frame label names to exclude. If None, no frame labels will be excluded.
        all_annotations: If True, all annotations will be downloaded, including the ones that are not set as closed/done.
    """

    def __init__(self,
                 project_name: str,
                 root: str | None = None,
                 auto_update: bool = True,
                 api_key: Optional[str] = None,
                 server_url: Optional[str] = None,
                 return_dicom: bool = False,
                 return_metainfo: bool = True,
                 return_frame_by_frame: bool = False,
                 return_annotations: bool = True,
                 # new parameters
                 return_segmentations: bool = True,
                 return_as_semantic_segmentation: bool = False,
                 image_transform: Callable[[torch.Tensor], Any] | None = None,
                 mask_transform: Callable[[torch.Tensor], Any] | None = None,
                 alb_transform: albumentations.BasicTransform | None = None,
                 semantic_seg_merge_strategy: Optional[Literal['union', 'intersection', 'mode']] = None,
                 include_unannotated: bool = True,
                 # filtering parameters
                 include_annotators: Optional[list[str]] = None,
                 exclude_annotators: Optional[list[str]] = None,
                 include_segmentation_names: Optional[list[str]] = None,
                 exclude_segmentation_names: Optional[list[str]] = None,
                 include_image_label_names: Optional[list[str]] = None,
                 exclude_image_label_names: Optional[list[str]] = None,
                 include_frame_label_names: Optional[list[str]] = None,
                 exclude_frame_label_names: Optional[list[str]] = None,
                 all_annotations: bool = False,
                 ):
        super().__init__(root=root,
                         project_name=project_name,
                         auto_update=auto_update,
                         api_key=api_key,
                         server_url=server_url,
                         return_dicom=return_dicom,
                         return_metainfo=return_metainfo,
                         return_frame_by_frame=return_frame_by_frame,
                         return_annotations=return_annotations,
                         include_unannotated=include_unannotated,
                         all_annotations=all_annotations,
                         include_annotators=include_annotators,
                         exclude_annotators=exclude_annotators,
                         include_segmentation_names=include_segmentation_names,
                         exclude_segmentation_names=exclude_segmentation_names,
                         include_image_label_names=include_image_label_names,
                         exclude_image_label_names=exclude_image_label_names,
                         include_frame_label_names=include_frame_label_names,
                         exclude_frame_label_names=exclude_frame_label_names
                         )
        self.return_segmentations = return_segmentations
        self.return_as_semantic_segmentation = return_as_semantic_segmentation
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.alb_transform = alb_transform
        if alb_transform is not None and return_frame_by_frame == False:
            # not supported yet
            raise NotImplementedError(
                "albumentations transform is not supported yet when return_frame_by_frame is False")
        self.semantic_seg_merge_strategy = semantic_seg_merge_strategy

        if return_segmentations == False and return_as_semantic_segmentation == True:
            raise ValueError("return_as_semantic_segmentation can only be True if return_segmentations is True")

        if semantic_seg_merge_strategy is not None and not return_as_semantic_segmentation:
            raise ValueError("semantic_seg_merge_strategy can only be used if return_as_semantic_segmentation is True")

    def _load_segmentations(self, annotations: list[Annotation], img_shape) -> tuple[dict[str, list], dict[str, list]]:
        """
        Load segmentations from annotations.

        Args:
            annotations: list of Annotation objects
            img_shape: shape of the image (#frames, C, H, W)

        Returns:
            tuple[dict[str, list], dict[str, list]]: a tuple of two dictionaries.
                The first dictionary is author -> list of #frames tensors, each tensor has shape (#instances_i, H, W).
                The second dictionary is author -> list of #frames segmentation labels (tensors).
        """
        segmentations = {}
        seg_labels = {}

        if self.return_frame_by_frame:
            assert len(img_shape) == 3, f"img_shape must have 3 dimensions, got {img_shape}"
            _, h, w = img_shape
            nframes = 1
        else:
            assert len(img_shape) == 4, f"img_shape must have 4 dimensions, got {img_shape}"
            nframes, _, h, w = img_shape

        # Load segmentation annotations
        for ann in annotations:
            if ann.type != 'segmentation':
                continue
            if ann.file is None:
                _LOGGER.warning(f"Segmentation annotation without file in annotations {ann}")
                continue
            author = ann.created_by

            segfilepath = ann.file  # png file
            segfilepath = os.path.join(self.dataset_dir, segfilepath)
            # FIXME: avoid enforcing resizing the mask
            seg = (Image.open(segfilepath)
                   .convert('L')
                   .resize((w, h), Image.NEAREST)
                   )
            seg = np.array(seg)

            seg = torch.from_numpy(seg)
            seg = seg == 255   # binary mask
            # map the segmentation label to the code
            seg_code = self.frame_lcodes['segmentation'][ann.name]
            if self.return_frame_by_frame:
                frame_index = 0
            else:
                frame_index = ann.index

            if author not in segmentations.keys():
                segmentations[author] = [None] * nframes
                seg_labels[author] = [None] * nframes
            author_segs = segmentations[author]
            author_labels = seg_labels[author]

            if author_segs[frame_index] is None:
                author_segs[frame_index] = []
                author_labels[frame_index] = []

            author_segs[frame_index].append(seg)
            author_labels[frame_index].append(seg_code)

        # convert to tensor
        for author in segmentations.keys():
            author_segs = segmentations[author]
            author_labels = seg_labels[author]
            for i in range(len(author_segs)):
                if author_segs[i] is not None:
                    author_segs[i] = torch.stack(author_segs[i])
                    author_labels[i] = torch.tensor(author_labels[i], dtype=torch.int32)
                else:
                    author_segs[i] = torch.zeros((0, h, w), dtype=torch.bool)
                    author_labels[i] = torch.zeros(0, dtype=torch.int32)

        return segmentations, seg_labels

    def _instanceseg2semanticseg(self,
                                 segmentations: List[Tensor],
                                 seg_labels: List[Tensor]) -> Tensor:
        """
        Convert instance segmentation to semantic segmentation.

        Args:
            segmentations: list of `n` tensors of shape (num_instances, H, W), where `n` is the number of frames.
            seg_labels: list of `n` tensors of shape (num_instances,), where `n` is the number of frames.

        Returns:
            Tensor: tensor of shape (n, num_labels, H, W), where `n` is the number of frames.
        """
        if segmentations is not None:
            if len(segmentations) != len(seg_labels):
                raise ValueError("segmentations and seg_labels must have the same length")

            h, w = segmentations[0].shape[1:]
            new_shape = (len(segmentations),
                         len(self.segmentation_labels_set)+1,  # +1 for background
                         h, w)
            new_segmentations = torch.zeros(new_shape, dtype=torch.uint8)
            # for each frame
            for i in range(len(segmentations)):
                # for each instance
                for j in range(len(segmentations[i])):
                    new_segmentations[i, seg_labels[i][j]] += segmentations[i][j]
            new_segmentations = new_segmentations > 0
            # pixels that are not in any segmentation are labeled as background
            new_segmentations[:, 0] = new_segmentations.sum(dim=1) == 0
            segmentations = new_segmentations.float()
        return segmentations

    def apply_semantic_seg_merge_strategy(self, segmentations: dict[str, Tensor],
                                          nframes: int,
                                          h, w) -> Tensor | dict[str, Tensor]:
        if self.semantic_seg_merge_strategy is None:
            return segmentations
        if len(segmentations) == 0:
            segmentations = torch.zeros((nframes, len(self.segmentation_labels_set)+1, h, w),
                                        dtype=torch.get_default_dtype())
            segmentations[:, 0, :, :] = 1  # background
            return segmentations
        if self.semantic_seg_merge_strategy == 'union':
            merged_segs = self._apply_semantic_seg_merge_strategy_union(segmentations)
        elif self.semantic_seg_merge_strategy == 'intersection':
            merged_segs = self._apply_semantic_seg_merge_strategy_intersection(segmentations)
        elif self.semantic_seg_merge_strategy == 'mode':
            merged_segs = self._apply_semantic_seg_merge_strategy_mode(segmentations)
        else:
            raise ValueError(f"Unknown semantic_seg_merge_strategy: {self.semantic_seg_merge_strategy}")
        return merged_segs.to(torch.get_default_dtype())

    def _apply_semantic_seg_merge_strategy_union(self, segmentations: Dict[str, torch.Tensor]) -> torch.Tensor:
        new_segmentations = torch.zeros_like(list(segmentations.values())[0])
        for seg in segmentations.values():
            new_segmentations += seg
        return new_segmentations.bool()

    def _apply_semantic_seg_merge_strategy_intersection(self, segmentations: Dict[str, torch.Tensor]) -> torch.Tensor:
        new_segmentations = torch.ones_like(list(segmentations.values())[0])
        for seg in segmentations.values():
            new_segmentations += seg
        return new_segmentations.bool()

    def _apply_semantic_seg_merge_strategy_mode(self, segmentations: Dict[str, torch.Tensor]) -> torch.Tensor:
        new_segmentations = torch.zeros_like(list(segmentations.values())[0])
        for seg in segmentations.values():
            new_segmentations += seg
        new_segmentations = new_segmentations >= len(segmentations) / 2
        return new_segmentations

    def __apply_alb_transform_segmentation(self,
                                           img: Tensor,
                                           segmentations: dict[str, list[Tensor]]
                                           ) -> tuple[np.ndarray, dict[str, list]]:
        all_masks_list = []
        num_masks = 0
        all_masks_key: dict[str, list] = {}
        for author_name, seglist in segmentations.items():
            all_masks_key[author_name] = []
            for i, seg in enumerate(seglist):
                if seg is not None:
                    all_masks_list.append(seg)
                    assert len(seg.shape) == 3, f"Segmentation must have 3 dimensions, got {seg.shape}"
                    all_masks_key[author_name].append([num_masks+j for j in range(seg.shape[0])])
                    num_masks += seg.shape[0]
                else:
                    all_masks_key[author_name].append(None)

        if len(all_masks_list) != 0:
            all_masks_list = torch.concatenate(all_masks_list).numpy().astype(np.uint8)
        else:
            all_masks_list = None  # np.empty((0,img.shape[-2], img.shape[-1]), dtype=np.uint8)

        augmented = self.alb_transform(image=img.numpy().transpose(1, 2, 0),
                                       masks=all_masks_list)

        # reconstruct the segmentations
        if all_masks_list is not None:
            all_masks = augmented['masks']  # shape: (num_masks, H, W)
        new_segmentations: dict[str, list] = {}
        for author_name, seglist in all_masks_key.items():
            new_segmentations[author_name] = []
            for i in range(len(seglist)):
                if seglist[i] is None:
                    new_segmentations[author_name].append(None)
                else:
                    masks_i = all_masks[seglist[i]]
                    masks_i = np.stack(masks_i)
                    new_segmentations[author_name].append(masks_i)

        return augmented['image'], new_segmentations

    def _seg_labels_to_names(self,
                             seg_labels: dict | list | None
                             ) -> dict | list | None:
        """
        Convert segmentation label codes to label names.

        Args:
            seg_labels: Segmentation labels in various formats:
                - dict[str, list[Tensor]]: author -> list of frame tensors with label codes
                - dict[str, Tensor]: author -> tensor with label codes
                - list[Tensor]: list of frame tensors with label codes
                - Tensor: tensor with label codes
                - None: when no segmentation labels are available

        Returns:
            Same structure as input but with label codes converted to label names.
            Returns None if input is None.
        """
        if seg_labels is None:
            return None

        code_to_name = self.segmentation_labels_set
        if isinstance(seg_labels, dict):
            # author -> list of frame tensors
            seg_names = {}
            for author, labels in seg_labels.items():
                if isinstance(labels, Tensor):
                    # single tensor for the author
                    seg_names[author] = [code_to_name[code.item()-1] for code in labels]
                elif isinstance(labels, list):
                    # list of frame tensors
                    seg_names[author] = [[code_to_name[code.item()-1] for code in frame_labels]
                                         for frame_labels in labels]
                else:
                    _LOGGER.warning(
                        f"Unexpected segmentation labels format for author {author}: {type(labels)}. Returning None")
                    return None
            return seg_names
        elif isinstance(seg_labels, list):
            # list of frame tensors
            return [[code_to_name[code.item()-1] for code in labels] for labels in seg_labels]

        _LOGGER.warning(f"Unexpected segmentation labels format: {type(seg_labels)}. Returning None")
        return None

    def __getitem__(self, index) -> dict[str, Any]:
        """
        Get the item at the given index.

        Args:
            index (int): Index of the item to return.

        Returns:
            dict[str, Any]: A dictionary with the following keys:

            * 'image' (Tensor): Tensor of shape (C, H, W) or (N, C, H, W), depending on `self.return_frame_by_frame`.
              If `self.return_as_semantic_segmentation` is True, the image is a tensor of shape (N, L, H, W) or (L, H, W),
              where `L` is the number of segmentation labels + 1 (background): ``L=len(self.segmentation_labels_set)+1``.
            * 'metainfo' (dict): Dictionary with metadata information.
            * 'segmentations' (dict[str, list[Tensor]] or dict[str,Tensor] or Tensor): Segmentation masks,
              depending on the configuration of parameters `self.return_segmentations`, `self.return_as_semantic_segmentation`, `self.return_frame_by_frame`, `self.semantic_seg_merge_strategy`.
            * 'seg_labels' (dict[str, list[Tensor]] or Tensor): Segmentation labels with the same length as `segmentations`.
            * 'frame_labels' (dict[str, Tensor]): Frame-level labels.
            * 'image_labels' (dict[str, Tensor]): Image-level labels.
        """
        item = super().__getitem__(index)
        img = item['image']
        metainfo = item['metainfo']
        annotations = item['annotations']

        has_transformed = False  # to check if albumentations transform was applied

        if self.image_transform is not None:
            img = self.image_transform(img)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        if img.ndim == 3:
            _, h, w = img.shape
            nframes = 1
        elif img.ndim == 4:
            nframes, _, h, w = img.shape
        else:
            raise ValueError(f"Image must have 3 or 4 dimensions, got {img.shape}")

        new_item = {
            'image': img,
            'metainfo': metainfo,
        }
        if 'dicom' in item:
            new_item['dicom'] = item['dicom']

        try:
            if self.return_segmentations:
                segmentations, seg_labels = self._load_segmentations(annotations, img.shape)
                # seg_labels can be dict[str, list[Tensor]]
                # apply mask transform
                if self.mask_transform is not None:
                    for seglist in segmentations.values():
                        for i, seg in enumerate(seglist):
                            if seg is not None:
                                seglist[i] = self.mask_transform(seg)

                if self.alb_transform is not None:
                    img, new_segmentations = self.__apply_alb_transform_segmentation(img, segmentations)
                    segmentations = new_segmentations
                    img = torch.from_numpy(img).permute(2, 0, 1)
                    new_item['image'] = img
                    has_transformed = True
                    # Update dimensions after transformation
                    if img.ndim == 3:
                        _, h, w = img.shape
                    elif img.ndim == 4:
                        nframes, _, h, w = img.shape

                if self.return_as_semantic_segmentation:
                    sem_segmentations: dict[str, torch.Tensor] = {}
                    for author in segmentations.keys():
                        sem_segmentations[author] = self._instanceseg2semanticseg(segmentations[author],
                                                                                  seg_labels[author])
                        segmentations[author] = None  # free memory
                    segmentations = self.apply_semantic_seg_merge_strategy(sem_segmentations,
                                                                           nframes,
                                                                           h, w)
                    # In semantic segmentation, seg_labels is not needed
                    seg_labels = None

                if self.return_frame_by_frame:
                    if isinstance(segmentations, dict):  # author->segmentations format
                        segmentations = {k: v[0] for k, v in segmentations.items()}
                        if seg_labels is not None:
                            seg_labels = {k: v[0] for k, v in seg_labels.items()}
                    else:
                        # segmentations is a tensor
                        segmentations = segmentations[0]
                        if seg_labels is not None and len(seg_labels) > 0:
                            seg_labels = seg_labels[0]
                new_item['segmentations'] = segmentations
                new_item['seg_labels'] = seg_labels
                # process seg_labels to convert from code to label names
                new_item['seg_labels_names'] = self._seg_labels_to_names(seg_labels)

        except Exception:
            _LOGGER.error(f'Error in loading/processing segmentations of {metainfo}')
            raise

        if self.alb_transform is not None and not has_transformed:
            # apply albumentations transform to the image
            augmented = self.alb_transform(image=img.numpy().transpose(1, 2, 0))
            img = torch.from_numpy(augmented['image']).permute(2, 0, 1)
            new_item['image'] = img

        framelabel_annotations = self._get_annotations_internal(annotations, type='label', scope='frame')
        framelabels = self._convert_labels_annotations(framelabel_annotations, num_frames=nframes)
        # framelabels.shape: (num_frames, num_labels)

        imagelabel_annotations = self._get_annotations_internal(annotations, type='label', scope='image')
        imagelabels = self._convert_labels_annotations(imagelabel_annotations)
        # imagelabels.shape: (num_labels,)

        new_item['frame_labels'] = framelabels
        new_item['image_labels'] = imagelabels

        # FIXME: deal with multiple annotators in instance segmentation

        return new_item

    def _convert_labels_annotations(self,
                                    annotations: list[Annotation],
                                    num_frames: int = None) -> dict[str, torch.Tensor]:
        """
        Converts the annotations, of the same type and scope, to tensor of shape (num_frames, num_labels)
        for each annotator.

        Args:
            annotations: list of Annotation objects
            num_frames: number of frames in the video

        Returns:
            dict[str, torch.Tensor]: dictionary of annotator_id -> tensor of shape (num_frames, num_labels)
        """
        if num_frames is None:
            labels_ret_size = (len(self.image_labels_set),)
            label2code = self.image_lcodes['multilabel']
            should_include_label = self._should_include_image_label
        else:
            labels_ret_size = (num_frames, len(self.frame_labels_set))
            label2code = self.frame_lcodes['multilabel']
            should_include_label = self._should_include_frame_label

        if num_frames is not None and num_frames > 1 and self.return_frame_by_frame:
            raise ValueError("num_frames must be 1 if return_frame_by_frame is True")

        frame_labels_byuser = {}  # defaultdict(lambda: torch.zeros(size=labels_ret_size, dtype=torch.int32))
        if len(annotations) == 0:
            return frame_labels_byuser
        for ann in annotations:
            user_id = ann.created_by

            frame_idx = ann.index

            if user_id not in frame_labels_byuser.keys():
                frame_labels_byuser[user_id] = torch.zeros(size=labels_ret_size, dtype=torch.int32)
            labels_onehot_i = frame_labels_byuser[user_id]
            code = label2code[ann.name]
            if frame_idx is None:
                labels_onehot_i[code] = 1
            else:
                if self.return_frame_by_frame:
                    labels_onehot_i[0, code] = 1
                else:
                    labels_onehot_i[frame_idx, code] = 1

        if self.return_frame_by_frame:
            for user_id, labels_onehot_i in frame_labels_byuser.items():
                frame_labels_byuser[user_id] = labels_onehot_i[0]
        return dict(frame_labels_byuser)

    def __repr__(self) -> str:
        super_repr = super().__repr__()
        body = []
        if self.image_transform is not None:
            body.append("Image transform:")
            body += [" " * 4 + line for line in repr(self.image_transform).split('\n')]
        if self.mask_transform is not None:
            body.append("Mask transform:")
            body += [" " * 4 + line for line in repr(self.mask_transform).split('\n')]
        if len(body) == 0:
            return super_repr
        lines = [" " * 4 + line for line in body]
        return super_repr + '\n' + "\n".join(lines)
