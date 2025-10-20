from typing import TypeAlias, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pydicom.dataset
    from PIL import Image
    import cv2
    from nibabel.filebasedimages import FileBasedImage as nib_FileBasedImage

# Type alias for imaging formats
ImagingData: TypeAlias = (
    Union[
        'pydicom.dataset.Dataset',
        'Image.Image',
        'cv2.VideoCapture',
        'nib_FileBasedImage'
    ]
)
