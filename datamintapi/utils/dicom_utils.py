import pydicom
from pydicom.uid import generate_uid
from typing import Sequence
import warnings
from copy import deepcopy
import logging
from pathlib import Path
from pydicom.misc import is_dicom as pydicom_is_dicom
from io import BytesIO
import os

_LOGGER = logging.getLogger(__name__)

CLEARED_STR = "CLEARED_BY_DATAMINT"


def anonymize_dicom(ds: pydicom.Dataset,
                    retain_codes: Sequence[tuple] = [],
                    copy=False,
                    ) -> pydicom.Dataset:
    """
    Anonymize a DICOM file by clearing all the specified DICOM tags
    according to the DICOM standard https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup55.pdf.
    This function will generate a new UID for the new DICOM file and clear the specified DICOM tags
    by replacing their values with "CLEARED_BY_DATAMINT".

    Args:
        ds: pydicom Dataset object.
        retain_codes: A list of DICOM tag codes to retain the value of.
        copy: If True, the function will return a copy of the input Dataset object.
            If False, the function will modify the input Dataset object in place.

    Returns:
        pydicom Dataset object with specified DICOM tags cleared
    """

    if copy:
        ds = deepcopy(ds)

    # Generate a new UID for the new DICOM file
    ds.SOPInstanceUID = generate_uid()
    if hasattr(ds, 'file_meta'):
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    # https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup55.pdf
    tags_to_clear = [
        (0x0008, 0x0014), (0x0008, 0x0050), (0x0008, 0x0080), (0x0008, 0x0081), (0x0008, 0x0090),
        (0x0008, 0x0092), (0x0008, 0x0094), (0x0008, 0x1010), (0x0008, 0x1030), (0x0008, 0x103E),
        (0x0008, 0x1040), (0x0008, 0x1048), (0x0008, 0x1050), (0x0008, 0x1060), (0x0008, 0x1070),
        (0x0008, 0x1080), (0x0008, 0x1155), (0x0008, 0x2111), (0x0010, 0x0010), (0x0010, 0x0020),
        (0x0010, 0x0030), (0x0010, 0x0032), (0x0010, 0x0040), (0x0010, 0x1000), (0x0010, 0x1001),
        (0x0010, 0x1010), (0x0010, 0x1020), (0x0010, 0x1030), (0x0010, 0x1090), (0x0010, 0x2160),
        (0x0010, 0x2180), (0x0010, 0x21B0), (0x0010, 0x4000), (0x0018, 0x1000), (0x0018, 0x1030),
        # (0x0020, 0x000D), (0x0020, 0x000E) StudyInstanceUID  and SeriesInstanceUID are retained
        (0x0020, 0x0010), (0x0020, 0x0052), (0x0020, 0x0200), (0x0020, 0x4000),
        (0x0040, 0x0275), (0x0040, 0xA730), (0x0088, 0x0140), (0x3006, 0x0024), (0x3006, 0x00C2)
    ]

    for code in retain_codes:
        if code in tags_to_clear:
            tags_to_clear.remove(code)

    # Clear the specified DICOM tags
    with warnings.catch_warnings():  # Supress UserWarning from pydicom
        warnings.filterwarnings("ignore", category=UserWarning, module='pydicom')
        for tag in tags_to_clear:
            if tag in ds:
                if tag == (0x0008, 0x0094):  # Phone number
                    ds[tag].value = "000-000-0000"
                # If tag is a floating point number, set it to 0.0
                elif ds[tag].VR in ['FL', 'FD', 'DS']:
                    ds[tag].value = 0
                elif ds[tag].VR == 'SQ':
                    del ds[tag]
                else:
                    try:
                        ds[tag].value = CLEARED_STR
                    except ValueError as e:
                        ds[tag].value = 0
    return ds


def is_dicom(f: str | Path | BytesIO) -> bool:
    if isinstance(f, BytesIO):
        fp = BytesIO(f.getbuffer())  # Avoid modifying the original BytesIO object
        fp.read(128)  # preamble

        return fp.read(4) == b"DICM"

    if isinstance(f, Path):
        f = str(f)
    if os.path.isdir(f):
        return False

    if f.endswith('.dcm') or f.endswith('.dicom'):
        return True

    # Check if the file has an extension
    if os.path.splitext(f)[1] != '':
        return False

    try:
        return pydicom_is_dicom(f)
    except FileNotFoundError as e:
        return None


def to_bytesio(ds: pydicom.Dataset, name: str) -> BytesIO:
    """
    Convert a pydicom Dataset object to BytesIO object.
    """
    dicom_bytes = BytesIO()
    pydicom.dcmwrite(dicom_bytes, ds)
    dicom_bytes.seek(0)
    dicom_bytes.name = name
    dicom_bytes.mode = 'rb'
    return dicom_bytes
