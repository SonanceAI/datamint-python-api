from .base_dataset import DatamintBaseDataset
from typing import Tuple, List, Optional, Callable, Any, Dict
import torch
import os
import numpy as np
import logging
from PIL import Image
from collections import defaultdict

_LOGGER = logging.getLogger(__name__)


class DatamintDataset(DatamintBaseDataset):
    def __init__(self,
                 root: str,
                 project_name: str,
                 dataset_name: str = None,
                 version=None,
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
                 ):
        super().__init__(root=root,
                         project_name=project_name,
                         dataset_name=dataset_name,
                         version=version,
                         auto_update=auto_update,
                         api_key=api_key,
                         server_url=server_url,
                         return_dicom=return_dicom,
                         return_metainfo=return_metainfo,
                         return_frame_by_frame=return_frame_by_frame,
                         return_annotations=return_annotations,
                         )
        self.return_segmentations = return_segmentations
        self.return_as_semantic_segmentation = return_as_semantic_segmentation
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        if return_segmentations == False and return_as_semantic_segmentation == True:
            raise ValueError("return_as_semantic_segmentation can only be True if return_segmentations is True")

    def _load_segmentations(self, annotations: List[Dict], img_shape) -> Tuple[Dict[str, List], Dict[str, List]]:
        segmentations = {}
        # segmentations = [None] * img_shape[0]
        # seg_labels = [None] * img_shape[0]
        seg_labels = {}
        # Load segmentation annotations
        for ann in annotations:
            if ann['type'] != 'segmentation':
                continue
            if 'file' not in ann:
                _LOGGER.warning(f"Segmentation annotation without file in annotations {ann}")
                continue
            author = ann['added_by']
            segfilepath = ann['file']  # png file
            segfilepath = os.path.join(self.dataset_dir, segfilepath)
            # FIXME: avoid enforcing resizing the mask
            seg = (Image.open(segfilepath)
                   .convert('L')
                   .resize((img_shape[2], img_shape[1]), Image.NEAREST)
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
                segmentations[author] = [None] * img_shape[0]
                seg_labels[author] = [None] * img_shape[0]
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
                    author_segs[i] = torch.zeros((0, img_shape[1], img_shape[2]), dtype=torch.bool)
                    author_labels[i] = torch.zeros(0, dtype=torch.int32)

        return segmentations, seg_labels

    def _instanceseg2semanticseg(self,
                                 segmentations: torch.Tensor,
                                 seg_labels: torch.Tensor) -> torch.Tensor:
        if segmentations is not None:
            new_shape = (len(segmentations),
                         len(self.segmentation_labels_set)+1,
                         segmentations.shape[1], segmentations.shape[2])
            new_segmentations = torch.zeros(new_shape, dtype=torch.uint8)
            for i in range(len(segmentations)):
                # for each frame
                new_segmentations[i, seg_labels[i]] += segmentations[i]
            new_segmentations = new_segmentations > 0
            # pixels that are not in any segmentation are labeled as background
            new_segmentations[:, 0] = new_segmentations.sum(dim=1) == 0
            segmentations = new_segmentations.float()
        return segmentations

    def __getitem__(self, index) -> Dict[str, Any]:
        item = super().__getitem__(index)
        img = item['image']
        metainfo = item['metainfo']
        annotations = item['annotations']

        new_item = {
            'image': img,
            'metainfo': metainfo,
        }

        if self.image_transform is not None:
            img = self.image_transform(img)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        if self.return_segmentations:
            segmentations, seg_labels = self._load_segmentations(annotations, img.shape)
            if self.return_as_semantic_segmentation:
                segmentations = self._instanceseg2semanticseg(segmentations, seg_labels)
            if self.mask_transform is not None:
                for i in range(len(segmentations)):
                    if segmentations[i] is not None:
                        segmentations[i] = self.mask_transform(segmentations[i])
            if self.return_frame_by_frame:
                segmentations = segmentations[0]
                seg_labels = seg_labels[0]
            new_item['segmentations'] = segmentations
            new_item['seg_labels'] = seg_labels

        framelabel_annotations = self._get_annotations_internal(annotations, type='label', scope='frame')
        framelabels = self._convert_framelabels_annotations(framelabel_annotations, num_frames=img.shape[0])
        new_item['frame_labels'] = framelabels

        # FIXME: deal with multiple annotators

        return new_item

    def _convert_framelabels_annotations(self, annotations: List[Dict], num_frames: int) -> Dict[str, torch.Tensor]:
        """
        Converts the annotations, of the same type and scope, to tensor of shape (num_frames, num_labels)
        for each annotator.

        Args:
            annotations: list of annotations
            num_frames: number of frames in the video

        Returns:
            Dict[torch.Tensor]: dictionary of annotator_id -> tensor of shape (num_frames, num_labels)
        """
        labels_ret_size = (num_frames, len(self.frame_labels_set))

        if len(annotations) == 0:
            return torch.zeros(size=labels_ret_size, dtype=torch.int32)

        label2code = self.frame_lcodes['multilabel']
        frame_labels_byuser = defaultdict(lambda: torch.zeros(size=labels_ret_size, dtype=torch.int32))
        for ann in annotations:
            user_id = ann['added_by']
            frame_idx = ann['index']

            labels_onehot_i = frame_labels_byuser[user_id]
            code = label2code[ann['name']]
            labels_onehot_i[frame_idx, code] = 1

        return dict(frame_labels_byuser)

    def __repr__(self) -> str:
        super_repr = super().__repr__()
        body = []
        if self.image_transform is not None:
            body += [repr(self.image_transform)]
        if self.mask_transform is not None:
            body += [repr(self.mask_transform)]
        if len(body) == 0:
            return super_repr
        lines = [" " * 4 + line for line in body]
        return super_repr + '\n' + "\n".join(lines)
