from typing import TYPE_CHECKING, Literal, TypeAlias, Union

if TYPE_CHECKING:
    import cv2
    import pydicom.dataset
    from nibabel.filebasedimages import FileBasedImage as nib_FileBasedImage
    from PIL import Image

# Type alias for imaging formats
ImagingData: TypeAlias = (
    Union[
        'pydicom.dataset.Dataset',
        'Image.Image',
        'cv2.VideoCapture',
        'nib_FileBasedImage'
    ]
)

CacheMode: TypeAlias = bool | Literal['loadonly']
