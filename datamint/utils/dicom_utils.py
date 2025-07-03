from pydicom.pixels import pixel_array
import pydicom
from pydicom.uid import generate_uid
from typing import Sequence, Generator, IO, TypeVar, Generic
import warnings
from copy import deepcopy
import logging
from pathlib import Path
from pydicom.misc import is_dicom as pydicom_is_dicom
from io import BytesIO
import os
import numpy as np
from collections import defaultdict
import uuid
import hashlib
from tqdm import tqdm

import pydicom.uid

_LOGGER = logging.getLogger(__name__)

CLEARED_STR = "CLEARED_BY_DATAMINT"

T = TypeVar('T')


class GeneratorWithLength(Generic[T]):
    def __init__(self, generator: Generator[T, None, None], length: int):
        self.generator = generator
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.generator

    def __next__(self) -> T:
        return next(self.generator)

    def close(self):
        self.generator.close()

    def throw(self, *args):
        return self.generator.throw(*args)

    def send(self, *args):
        return self.generator.send(*args)


class TokenMapper:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_token(self, tag: tuple, value: str, simple_id=False) -> str:
        """Get a consistent token for a given tag and value pair."""
        if value is None or value == CLEARED_STR:
            return CLEARED_STR

        # Use a hash function to generate a consistent token
        token = hashlib.md5(f"{tag}{value}{self.seed}".encode()).hexdigest()
        if simple_id:
            return token
        return generate_uid(entropy_srcs=['DATAMINT', token])


_TOKEN_MAPPER = TokenMapper()


def anonymize_dicom(ds: pydicom.Dataset,
                    retain_codes: Sequence[tuple] = [],
                    copy=False,
                    token_mapper: TokenMapper = None) -> pydicom.Dataset:
    """
    Anonymize a DICOM file by clearing all the specified DICOM tags
    according to the DICOM standard https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup55.pdf.
    This function will generate a new UID for the new DICOM file and clear the specified DICOM tags
    with consistent tokens for related identifiers.

    Args:
        ds: pydicom Dataset object.
        retain_codes: A list of DICOM tag codes to retain the value of.
        copy: If True, the function will return a copy of the input Dataset object.
        token_mapper: TokenMapper instance to maintain consistent tokens across calls.
            If None, uses a global instance.

    Returns:
        pydicom Dataset object with specified DICOM tags cleared
    """
    if copy:
        ds = deepcopy(ds)

    if token_mapper is None:
        token_mapper = _TOKEN_MAPPER

    # https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup55.pdf
    tags_to_clear = [
        (0x0008, 0x0014), (0x0008, 0x0050), (0x0008, 0x0080), (0x0008, 0x0081), (0x0008, 0x0090),
        (0x0008, 0x0092), (0x0008, 0x0094), (0x0008, 0x1010), (0x0008, 0x1030), (0x0008, 0x103E),
        (0x0008, 0x1040), (0x0008, 0x1048), (0x0008, 0x1050), (0x0008, 0x1060), (0x0008, 0x1070),
        (0x0008, 0x1080), (0x0008, 0x1155), (0x0008, 0x2111), (0x0010, 0x0010), (0x0010, 0x0020),
        (0x0010, 0x0030), (0x0010, 0x0032), (0x0010, 0x0040), (0x0010, 0x1000), (0x0010, 0x1001),
        (0x0010, 0x1010), (0x0010, 0x1020), (0x0010, 0x1030), (0x0010, 0x1090), (0x0010, 0x2160),
        (0x0010, 0x2180), (0x0010, 0x21B0), (0x0010, 0x4000), (0x0018, 0x1000), (0x0018, 0x1030),
        (0x0020, 0x000D), (0x0020, 0x000E),  # StudyInstanceUID  and SeriesInstanceUID
        (0x0020, 0x0010), (0x0020, 0x0052), (0x0020, 0x0200), (0x0020, 0x4000), (0x0008, 0x0018),
        (0x0040, 0x0275), (0x0040, 0xA730), (0x0088, 0x0140), (0x3006, 0x0024), (0x3006, 0x00C2)
    ]

    # Frame of Reference UID, Series Instance UID, Concatenation UID, and Instance UID, and StudyInstanceUID are converted to new UIDs
    uid_tags = [(0x0020, 0x0052), (0x0020, 0x000E), (0x0020, 0x9161),
                (0x0010, 0x0020), (0x0008, 0x0018), (0x0020, 0x000D)]
    simple_id_tags = [(0x0010, 0x0020)]  # Patient ID

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
                    if tag in uid_tags:
                        try:
                            # Use consistent token mapping for identifiers
                            original_value = ds[tag].value
                            ds[tag].value = token_mapper.get_token(tag, original_value, simple_id=tag in simple_id_tags)
                            tag_name = pydicom.datadict.keyword_for_tag(tag)
                        except ValueError as e:
                            ds[tag].value = CLEARED_STR
                    else:
                        ds[tag].value = CLEARED_STR
    if hasattr(ds, 'file_meta') and hasattr(ds, 'SOPInstanceUID'):
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
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

    fname = f.lower()
    if fname.endswith('.dcm') or fname.endswith('.dicom'):
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


def load_image_normalized(dicom: pydicom.Dataset, index: int = None) -> np.ndarray:
    """
    Normalizes the shape of an array of images to (n, c, y, x)=(#slices, #channels, height, width).
    It uses dicom.Rows, dicom.Columns, and other information to determine the shape.

    Args:
        dicom: A dicom with images of varying shapes.

    Returns:
        A numpy array of shape (n, c, y, x)=(#slices, #channels, height, width).
    """
    n = dicom.get('NumberOfFrames', 1)
    if index is None:
        images = dicom.pixel_array
    else:
        if index is not None and index >= n:
            raise ValueError(f"Index {index} is out of bounds. The number of frames is {n}.")
        images = pixel_array(dicom, index=index)
        n = 1
    shape = images.shape

    c = dicom.get('SamplesPerPixel')

    # x=width, y=height
    if images.ndim == 2:
        # Single grayscale image (y, x)
        # Reshape to (1, 1, y, x)
        return images.reshape((1, 1) + images.shape)
    elif images.ndim == 3:
        # (n, y, x) or (y, x, c)
        if shape[0] == 1 or (n is not None and n > 1):
            # (n, y, x)
            return images.reshape(shape[0], 1, shape[1], shape[2])
        if shape[2] in (1, 3, 4) or (c is not None and c > 1):
            # (y, x, c)
            images = images.transpose(2, 0, 1)
            return images.reshape(1, *images.shape)
    elif images.ndim == 4:
        if shape[3] == c or shape[3] in (1, 3, 4) or (c is not None and c > 1):
            # (n, y, x, c) -> (n, c, y, x)
            return images.transpose(0, 3, 1, 2)

    raise ValueError(f"Unsupported DICOM normalization with shape: {shape}, SamplesPerPixel: {c}, NumberOfFrames: {n}")


def assemble_dicoms(files_path: list[str | IO],
                    return_as_IO: bool = False) -> GeneratorWithLength[pydicom.Dataset | IO]:
    """
    Assemble multiple DICOM files into a single multi-frame DICOM file.
    This function will merge the pixel data of the DICOM files and generate a new DICOM file with the combined pixel data.

    Args:
        files_path: A list of file paths to the DICOM files to be merged.

    Returns:
        A generator that yields the merged DICOM files.
    """
    dicoms_map = defaultdict(list)
    
    for file_path in tqdm(files_path, desc="Reading DICOMs metadata", unit="file"):
        dicom = pydicom.dcmread(file_path,
                                specific_tags=['SeriesInstanceUID', 'InstanceNumber', 'Rows', 'Columns'])
        series_uid = dicom.get('SeriesInstanceUID', None)
        if series_uid is None:
            # generate a random uid
            series_uid = pydicom.uid.generate_uid()
        instance_number = dicom.get('InstanceNumber', 0)
        rows = dicom.get('Rows', None)
        columns = dicom.get('Columns', None)
        dicoms_map[series_uid].append((instance_number, file_path, rows, columns))
        if hasattr(file_path, "seek"):
            file_path.seek(0)
    
    # Validate that all DICOMs with the same SeriesInstanceUID have matching dimensions
    for series_uid, dicom_list in dicoms_map.items():
        if len(dicom_list) <= 1:
            continue
            
        # Get dimensions from first DICOM
        first_rows = dicom_list[0][2]
        first_columns = dicom_list[0][3]
        
        # Check all other DICOMs have the same dimensions
        for instance_number, file_path, rows, columns in dicom_list:
            if rows != first_rows or columns != first_columns:
                msg = (
                    f"Dimension mismatch in SeriesInstanceUID {series_uid}: "
                    f"Expected {first_rows}x{first_columns}, got {rows}x{columns} "
                    f"for file {file_path} and {dicom_list[0][1]}"
                )
                _LOGGER.error(msg)
                raise ValueError(msg)

    # filter out the two last elements of the tuple (rows, columns)
    dicoms_map = {fr_uid: [(instance_number, file_path) for instance_number, file_path, _, _ in dicoms]
                  for fr_uid, dicoms in dicoms_map.items()}
    
    gen = _generate_merged_dicoms(dicoms_map, return_as_IO=return_as_IO)
    return GeneratorWithLength(gen, len(dicoms_map))


def _create_multiframe_attributes(merged_ds: pydicom.Dataset,
                                  all_dicoms: list[pydicom.Dataset]) -> pydicom.Dataset:
    ### Shared Functional Groups Sequence ###
    shared_seq_dataset = pydicom.dataset.Dataset()

    # check if pixel spacing or spacing between slices are equal for all dicoms
    pixel_spacing = merged_ds.get('PixelSpacing', None)
    all_pixel_spacing_equal = all(ds.get('PixelSpacing', None) == pixel_spacing
                                  for ds in all_dicoms)
    spacing_between_slices = merged_ds.get('SpacingBetweenSlices', None)
    all_spacing_b_slices_equal = all(ds.get('SpacingBetweenSlices', None) == spacing_between_slices
                                     for ds in all_dicoms)

    # if they are equal, add them to the shared functional groups sequence
    if (pixel_spacing is not None and all_pixel_spacing_equal) or (spacing_between_slices is not None and all_spacing_b_slices_equal):
        pixel_measure = pydicom.dataset.Dataset()
        if pixel_spacing is not None:
            pixel_measure.PixelSpacing = pixel_spacing
        if spacing_between_slices is not None:
            pixel_measure.SpacingBetweenSlices = spacing_between_slices
        pixel_measures_seq = pydicom.Sequence([pixel_measure])
        shared_seq_dataset.PixelMeasuresSequence = pixel_measures_seq

    if len(shared_seq_dataset) > 0:
        shared_seq = pydicom.Sequence([shared_seq_dataset])
        merged_ds.SharedFunctionalGroupsSequence = shared_seq
    #######

    ### Per-Frame Functional Groups Sequence ###
    perframe_seq_list = []
    for ds in all_dicoms:
        per_frame_dataset = pydicom.dataset.Dataset()  # root dataset for each frame
        pos_dataset = pydicom.dataset.Dataset()
        orient_dataset = pydicom.dataset.Dataset()
        pixel_measure = pydicom.dataset.Dataset()
        framenumber_dataset = pydicom.dataset.Dataset()

        if 'ImagePositionPatient' in ds:
            pos_dataset.ImagePositionPatient = ds.ImagePositionPatient
        if 'ImageOrientationPatient' in ds:
            orient_dataset.ImageOrientationPatient = ds.ImageOrientationPatient
        if 'PixelSpacing' in ds and all_pixel_spacing_equal == False:
            pixel_measure.PixelSpacing = ds.PixelSpacing
        if 'SpacingBetweenSlices' in ds and all_spacing_b_slices_equal == False:
            pixel_measure.SpacingBetweenSlices = ds.SpacingBetweenSlices

        # Add datasets to the per-frame dataset
        per_frame_dataset.PlanePositionSequence = pydicom.Sequence([pos_dataset])
        per_frame_dataset.PlaneOrientationSequence = pydicom.Sequence([orient_dataset])
        per_frame_dataset.PixelMeasuresSequence = pydicom.Sequence([pixel_measure])
        per_frame_dataset.FrameContentSequence = pydicom.Sequence([framenumber_dataset])

        perframe_seq_list.append(per_frame_dataset)
    if len(perframe_seq_list[0]) > 0:
        perframe_seq = pydicom.Sequence(perframe_seq_list)
        merged_ds.PerFrameFunctionalGroupsSequence = perframe_seq
        merged_ds.FrameIncrementPointer = (0x5200, 0x9230)

    return merged_ds


def _generate_dicom_name(ds: pydicom.Dataset) -> str:
    """
    Generate a meaningful name for a DICOM dataset using its attributes.

    Args:
        ds: pydicom Dataset object

    Returns:
        A string containing a descriptive name with .dcm extension
    """
    components = []

    # if hasattr(ds, 'filename'):
    #     components.append(os.path.basename(ds.filename))
    if hasattr(ds, 'SeriesDescription'):
        components.append(ds.SeriesDescription)
    if len(components) == 0 and hasattr(ds, 'SeriesNumber'):
        components.append(f"ser{ds.SeriesNumber}")
    if hasattr(ds, 'StudyDescription'):
        components.append(ds.StudyDescription)
    elif hasattr(ds, 'StudyID'):
        components.append(ds.StudyID)

    # Join components and add extension
    if len(components) > 0:
        description = "_".join(str(x) for x in components) + ".dcm"
        # Clean description - remove special chars and spaces
        description = "".join(c if c.isalnum() else "_" for c in description)
        if len(description) > 0:
            return description

    if hasattr(ds, 'SeriesInstanceUID'):
        return ds.SeriesInstanceUID + ".dcm"

    # Fallback to generic name if no attributes found
    return ds.filename if hasattr(ds, 'filename') else f"merged_dicom_{uuid.uuid4()}.dcm"


def _generate_merged_dicoms(dicoms_map: dict[str, list],
                            return_as_IO: bool = False) -> Generator[pydicom.Dataset, None, None]:
    for _, dicoms in dicoms_map.items():
        dicoms.sort(key=lambda x: x[0])
        files_path = [file_path for _, file_path in dicoms]

        all_dicoms = [pydicom.dcmread(file_path) for file_path in files_path]

        # Use the first dicom as a template
        merged_dicom = all_dicoms[0]

        # Combine pixel data
        pixel_arrays = np.stack([ds.pixel_array for ds in all_dicoms], axis=0)

        # Update the merged dicom
        merged_dicom.PixelData = pixel_arrays.tobytes()
        merged_dicom.NumberOfFrames = len(pixel_arrays)  # Set number of frames
        merged_dicom.SOPInstanceUID = pydicom.uid.generate_uid()  # Generate new SOP Instance UID
        # Removed deprecated attributes and set Transfer Syntax UID instead:
        merged_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # Free up memory
        for ds in all_dicoms[1:]:
            del ds.PixelData

        # create multi-frame attributes
        # check if FramTime is equal for all dicoms
        frame_time = merged_dicom.get('FrameTime', None)
        all_frame_time_equal = all(ds.get('FrameTime', None) == frame_time for ds in all_dicoms)
        if frame_time is not None and all_frame_time_equal:
            merged_dicom.FrameTime = frame_time  # (0x0018,0x1063)
            merged_dicom.FrameIncrementPointer = (0x0018, 0x1063)  # points to 'FrameTime'
        else:
            # TODO: Sometimes FrameTime is present but not equal for all dicoms. In this case, check out 'FrameTimeVector'.
            merged_dicom = _create_multiframe_attributes(merged_dicom, all_dicoms)

        # Remove tags of single frame dicoms
        for attr in ['ImagePositionPatient', 'SliceLocation', 'ImageOrientationPatient',
                     'PixelSpacing', 'SpacingBetweenSlices', 'InstanceNumber']:
            if hasattr(merged_dicom, attr):
                delattr(merged_dicom, attr)

        if return_as_IO:
            name = _generate_dicom_name(merged_dicom)
            yield to_bytesio(merged_dicom, name=name)
        else:
            yield merged_dicom


"""
- The Slice Location (0020,1041) is usually a derived attribute,
typically computed from Image Position (Patient) (0020,0032)
"""


def get_space_between_slices(ds: pydicom.Dataset) -> float:
    """
    Get the space between slices from a DICOM dataset.

    Parameters:
        ds (pydicom.Dataset): The DICOM dataset containing image metadata.

    Returns:
        float: Space between slices in millimeters.
    """
    # Get the Spacing Between Slices attribute
    if 'SpacingBetweenSlices' in ds:
        return ds.SpacingBetweenSlices

    if 'SharedFunctionalGroupsSequence' in ds:
        shared_group = ds.SharedFunctionalGroupsSequence[0]
        if 'PixelMeasuresSequence' in shared_group and 'SpacingBetweenSlices' in shared_group.PixelMeasuresSequence[0]:
            return shared_group.PixelMeasuresSequence[0].SpacingBetweenSlices

    if 'SliceThickness' in ds:
        return ds.SliceThickness

    return 1.0  # Default value if not found


def get_image_orientation(ds: pydicom.Dataset, slice_index: int) -> np.ndarray:
    """
    Get the image orientation from a DICOM dataset.

    Parameters:
        ds (pydicom.Dataset): The DICOM dataset containing image metadata.

    Returns:
        numpy.ndarray: Image orientation (X, Y, Z) for the specified slice.
    """
    # Get the Image Orientation Patient attribute
    if 'ImageOrientationPatient' in ds:
        return ds.ImageOrientationPatient

    if 'PerFrameFunctionalGroupsSequence' in ds:
        if 'PlaneOrientationSequence' in ds.PerFrameFunctionalGroupsSequence[slice_index]:
            return ds.PerFrameFunctionalGroupsSequence[slice_index].PlaneOrientationSequence[0].ImageOrientationPatient

    if 'SharedFunctionalGroupsSequence' in ds:
        return ds.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient

    raise ValueError("ImageOrientationPatient not found in DICOM dataset.")


def get_slice_orientation(ds: pydicom.Dataset, slice_index: int) -> np.ndarray:
    """
    Get the slice orientation from a DICOM dataset.

    Parameters:
        ds (pydicom.Dataset): The DICOM dataset containing image metadata.
        slice_index (int): 0-based index of the slice in the 3D volume. This is the `InstanceNumber-1`.

    Returns:
        numpy.ndarray: Slice orientation (X, Y, Z) for the specified slice.
    """
    # Get the Image Orientation Patient attribute

    x_orient, y_orient = np.array(get_image_orientation(ds, slice_index), dtype=np.float64).reshape(2, 3)
    # compute the normal vector of the slice
    slice_orient = np.cross(x_orient, y_orient)
    # normalize the vector to space_between_slices
    space_between_slices = get_space_between_slices(ds)
    slice_orient = slice_orient / np.linalg.norm(slice_orient) * space_between_slices

    return slice_orient


def _get_instance_number(ds: pydicom.Dataset, slice_index: int | None = None) -> int:
    if slice_index is None:
        if 'InstanceNumber' in ds and ds.InstanceNumber is not None:
            return ds.InstanceNumber
        elif 'NumberOfFrames' in ds and ds.NumberOfFrames == 1:
            return 0
        else:
            raise ValueError("Slice index is required for multi-frame images.")
    else:
        if slice_index < 0:
            raise ValueError("Slice index must be a non-negative integer.")
        if 'NumberOfFrames' in ds and slice_index >= ds.NumberOfFrames:
            _LOGGER.warning(f"Slice index {slice_index} exceeds number of frames {ds.NumberOfFrames}.")
        root_instance_number = ds.get('InstanceNumber', 1)
        if root_instance_number is None:
            root_instance_number = 1
        return root_instance_number + slice_index


def get_image_position(ds: pydicom.Dataset,
                       slice_index: int | None = None) -> np.ndarray:
    """
    Get the image position for a specific slice in a DICOM dataset.

    Parameters:
        ds (pydicom.Dataset): The DICOM dataset containing image metadata.
        slice_index (int): Index of the slice in the 3D volume.

    Returns:
        numpy.ndarray: Image position (X, Y, Z) for the specified slice.
    """

    instance_number = _get_instance_number(ds, slice_index)

    if 'PerFrameFunctionalGroupsSequence' in ds:
        if slice_index is not None:
            frame_groups = ds.PerFrameFunctionalGroupsSequence[slice_index]
            if 'PlanePositionSequence' in frame_groups and 'ImagePositionPatient' in frame_groups.PlanePositionSequence[0]:
                return frame_groups.PlanePositionSequence[0].ImagePositionPatient
        else:
            logging.warning("PerFrameFunctionalGroupsSequence is available, but slice_index is not provided.")

    # Get the Image Position Patient attribute
    if 'ImagePositionPatient' in ds:
        if 'SliceLocation' in ds:
            _LOGGER.debug("SliceLocation attribute is available, but not accounted for in calculation.")
        x = np.array(ds.ImagePositionPatient, dtype=np.float64)
        sc_orient = get_slice_orientation(ds, slice_index)
        return x + sc_orient*(instance_number-ds.get('InstanceNumber', 1))

    raise ValueError("ImagePositionPatient not found in DICOM dataset.")


def get_pixel_spacing(ds: pydicom.Dataset, slice_index: int) -> np.ndarray:
    """
    Get the pixel spacing from a DICOM dataset.

    Parameters:
        ds (pydicom.Dataset): The DICOM dataset containing image metadata.
        slice_index (int): Index of the slice in the 3D volume.

    Returns:
        numpy.ndarray: Pixel spacing (X, Y) for the specified slice.
    """
    # Get the Pixel Spacing attribute
    if 'PixelSpacing' in ds:
        return np.array(ds.PixelSpacing, dtype=np.float64)

    if 'PerFrameFunctionalGroupsSequence' in ds:
        if 'PixelMeasuresSequence' in ds.PerFrameFunctionalGroupsSequence[slice_index]:
            return ds.PerFrameFunctionalGroupsSequence[slice_index].PixelMeasuresSequence[0].PixelSpacing

    if 'SharedFunctionalGroupsSequence' in ds:
        if 'PixelMeasuresSequence' in ds.SharedFunctionalGroupsSequence[0]:
            return ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing

    raise ValueError("PixelSpacing not found in DICOM dataset.")


def pixel_to_patient(ds: pydicom.Dataset,
                     pixel_x, pixel_y,
                     slice_index: int | None = None,
                     instance_number: int | None = None) -> np.ndarray:
    """
    Convert pixel coordinates (pixel_x, pixel_y) to patient coordinates in DICOM.

    Parameters:
        ds (pydicom.Dataset): The DICOM dataset containing image metadata.
        pixel_x (float): X coordinate in pixel space.
        pixel_y (float): Y coordinate in pixel space.
        slice_index (int): Index of the slice of the `ds.pixel_array`.
        instance_number (int): Instance number of the slice in the 3D volume.


    Returns:
        numpy.ndarray: Patient coordinates (X, Y, Z).
    """

    # - image_position is the origin of the image in patient coordinates (ImagePositionPatient)
    # - row_vector and col_vector are the direction cosines from ImageOrientationPatient
    # - pixel_spacing is the physical distance between the centers of adjacent pixels

    if slice_index is not None and instance_number is not None:
        raise ValueError("Either slice_index or instance_number should be provided, not both.")

    if slice_index is None:
        if instance_number is None:
            instance_number = _get_instance_number(ds)
        root_instance_number = ds.get('InstanceNumber', 1)
        if root_instance_number is None:
            root_instance_number = 1
        slice_index = instance_number - root_instance_number

    # Get required DICOM attributes
    image_position = np.array(get_image_position(ds, slice_index), dtype=np.float64)
    image_orientation = np.array(get_image_orientation(ds, slice_index), dtype=np.float64).reshape(2, 3)
    # image_position = np.array(ds.ImagePositionPatient, dtype=np.float64)  # (0020,0032)
    # image_orientation = np.array(ds.ImageOrientationPatient, dtype=np.float64).reshape(2, 3)  # (0020,0037)
    # pixel_spacing = np.array(ds.PixelSpacing, dtype=np.float64)  # (0028,0030)
    pixel_spacing = np.array(get_pixel_spacing(ds, slice_index), dtype=np.float64)  # (0028,0030)

    # Compute row and column vectors from image orientation
    row_vector = image_orientation[0]
    col_vector = image_orientation[1]

    # Compute patient coordinates
    patient_coords = image_position + pixel_x * pixel_spacing[0] * row_vector + pixel_y * pixel_spacing[1] * col_vector

    return patient_coords
