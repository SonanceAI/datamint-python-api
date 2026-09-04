from typing import TYPE_CHECKING, Literal, TypeAlias, Union

if TYPE_CHECKING:
    import numpy as np
    import pydicom.dataset
    from nibabel.filebasedimages import FileBasedImage as nib_FileBasedImage
    from PIL import Image

# Type alias for imaging formats
ImagingData: TypeAlias = (
    Union[
        'pydicom.dataset.Dataset',
        'Image.Image',
        'np.ndarray',
        'nib_FileBasedImage'
    ]
)

CacheMode: TypeAlias = bool | Literal['loadonly']
