"""
VideoDataset - Dataset for video medical data.

Handles video files (MP4, AVI, etc.) and multi-frame DICOM data from
modalities like ultrasound (US), angiography (XA), and fluoroscopy (RF).
"""
import logging
from typing import TYPE_CHECKING

from .multiframe_dataset import MultiFrameDataset

if TYPE_CHECKING:
    from .sliced_video_dataset import SlicedVideoDataset

_LOGGER = logging.getLogger(__name__)


class VideoDataset(MultiFrameDataset):
    """Dataset for video medical data.

    Each item is a full video with shape ``(C, N, H, W)`` where ``N`` is the
    number of frames. Inherits multi-frame loading and augmentation from
    :class:`MultiFrameDataset`.

    Supports video files (MP4, AVI, MOV) and multi-frame DICOM from temporal
    modalities (ultrasound, angiography, fluoroscopy).

    Example::

        ds = VideoDataset(project='my_ultrasound_project')
        item = ds[0]
        print(item['image'].shape)  # (C, N, H, W)

        # Iterate frame-by-frame
        frame_ds = ds.frame_by_frame()
        print(frame_ds[0]['image'].shape)  # (C, H, W)
    """

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"VideoDataset\n{base}"

    def frame_by_frame(self) -> 'SlicedVideoDataset':
        """Create a 2D dataset iterating over individual video frames.

        Each video is expanded into ``N`` individual frames. The returned
        dataset yields 2D items with shape ``(C, H, W)`` instead of
        ``(C, N, H, W)``.

        Parsed frames are cached to disk as gzip-compressed ``.npy.gz`` files.

        Returns:
            A :class:`SlicedVideoDataset` that iterates over individual frames.

        Example::

            vid_ds = VideoDataset(project='my_ultrasound_project')
            frame_ds = vid_ds.frame_by_frame()
            print(len(frame_ds))  # total number of frames across all videos
            item = frame_ds[0]
            print(item['image'].shape)  # (C, H, W)
        """
        from .sliced_video_dataset import SlicedVideoDataset

        return SlicedVideoDataset.from_dataset(parent_dataset=self)