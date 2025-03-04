from .base_dataset import DatamintBaseDataset
from typing import List, Optional, Callable, Any, Dict, Literal, Union
import torch
from torch import Tensor
import os
import numpy as np
import logging
from PIL import Image
from collections import defaultdict

_LOGGER = logging.getLogger(__name__)


class DatamintDataset(DatamintBaseDataset):
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
                 image_transform: Callable[[torch.Tensor], Any] = None,
                 mask_transform: Callable[[torch.Tensor], Any] = None,
                 semantic_seg_merge_strategy: Optional[Literal['union', 'intersection', 'mode']] = None,
                 discard_without_annotations: bool = False,
                 # annotator filtering parameters
                 include_annotators: Optional[List[str]] = None,
                 exclude_annotators: Optional[List[str]] = None,
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
                         discard_without_annotations=discard_without_annotations,
                         )
        self.return_segmentations = return_segmentations
        self.return_as_semantic_segmentation = return_as_semantic_segmentation
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.semantic_seg_merge_strategy = semantic_seg_merge_strategy
        self.include_annotators = include_annotators
        self.exclude_annotators = exclude_annotators

        if return_segmentations == False and return_as_semantic_segmentation == True:
            raise ValueError("return_as_semantic_segmentation can only be True if return_segmentations is True")

        if semantic_seg_merge_strategy is not None and not return_as_semantic_segmentation:
            raise ValueError("semantic_seg_merge_strategy can only be used if return_as_semantic_segmentation is True")

        if include_annotators is not None and exclude_annotators is not None:
            raise ValueError("Cannot set both include_annotators and exclude_annotators at the same time")

    def _should_include_annotator(self, annotator_id: str) -> bool:
        """
        Check if an annotator should be included based on the filtering settings.

        Args:
            annotator_id: The ID of the annotator to check

        Returns:
            bool: True if the annotator should be included, False otherwise
        """
        if self.include_annotators is not None:
            return annotator_id in self.include_annotators
        if self.exclude_annotators is not None:
            return annotator_id not in self.exclude_annotators
        return True

    def _load_segmentations(self, annotations: list[dict], img_shape) -> tuple[dict[str, list], dict[str, list]]:
        """
        Load segmentations from annotations.

        Args:
            annotations: list of annotations. Each annotation is a dictionary with keys 'type', 'file', 'added_by', 'name', 'index'.
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
            if ann['type'] != 'segmentation':
                continue
            if 'file' not in ann:
                _LOGGER.warning(f"Segmentation annotation without file in annotations {ann}")
                continue
            author = ann['added_by']

            # Skip if annotator should be filtered out
            if not self._should_include_annotator(author):
                continue

            segfilepath = ann['file']  # png file
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
            seg_code = self.frame_lcodes['segmentation'][ann['name']]
            if self.return_frame_by_frame:
                frame_index = 0
            else:
                frame_index = ann['index']

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

        if self.return_segmentations:
            segmentations, seg_labels = self._load_segmentations(annotations, img.shape)
            # apply mask transform
            if self.mask_transform is not None:
                for seglist in segmentations.values():
                    for i, seg in enumerate(seglist):
                        if seg is not None:
                            seglist[i] = self.mask_transform(seg)

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
                                    annotations: list[dict],
                                    num_frames: int = None) -> dict[str, torch.Tensor]:
        """
        Converts the annotations, of the same type and scope, to tensor of shape (num_frames, num_labels)
        for each annotator.

        Args:
            annotations: list of annotations
            num_frames: number of frames in the video

        Returns:
            Dict[torch.Tensor]: dictionary of annotator_id -> tensor of shape (num_frames, num_labels)
        """
        if num_frames is None:
            labels_ret_size = (len(self.image_labels_set),)
            label2code = self.image_lcodes['multilabel']
        else:
            labels_ret_size = (num_frames, len(self.frame_labels_set))
            label2code = self.frame_lcodes['multilabel']

        if num_frames is not None and num_frames > 1 and self.return_frame_by_frame:
            raise ValueError("num_frames must be 1 if return_frame_by_frame is True")

        frame_labels_byuser = {} #defaultdict(lambda: torch.zeros(size=labels_ret_size, dtype=torch.int32))
        if len(annotations) == 0:
            return frame_labels_byuser
        for ann in annotations:
            user_id = ann['added_by']

            # Skip if annotator should be filtered out
            if not self._should_include_annotator(user_id):
                continue

            frame_idx = ann.get('index', None)

            if user_id not in frame_labels_byuser.keys():
                frame_labels_byuser[user_id] = torch.zeros(size=labels_ret_size, dtype=torch.int32)
            labels_onehot_i = frame_labels_byuser[user_id]
            code = label2code[ann['name']]
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
        """
        Example:
            .. code-block:: python

                print(dataset)

            Output:

            .. code-block:: text

                Dataset DatamintDataset
                    Number of datapoints: 3
                    Root location: /home/user/.datamint/datasets

        """
        super_repr = super().__repr__()
        body = []
        if self.image_transform is not None:
            body += [repr(self.image_transform)]
        if self.mask_transform is not None:
            body += [repr(self.mask_transform)]
        if self.include_annotators is not None:
            body += [f"Including only annotators: {self.include_annotators}"]
        if self.exclude_annotators is not None:
            body += [f"Excluding annotators: {self.exclude_annotators}"]
        if len(body) == 0:
            return super_repr
        lines = [" " * 4 + line for line in body]
        return super_repr + '\n' + "\n".join(lines)
