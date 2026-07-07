import logging

import numpy as np
import nibabel as nib
from tqdm.auto import tqdm

from datamint import Api
from datamint.entities import Project

from . import _common, _download

_LOGGER = logging.getLogger(__name__)

_SYNAPSE_URL = 'https://www.kaggle.com/api/v1/datasets/download/dogcdt/synapse'
_DATASET_NAME = 'Synapse Segmentation Example'
_DESCRIPTION = ('Synapse Multi-Organ CT 3D segmentation example dataset, '
               'auto-populated by datamint.examples.synapse_dataset.')

_SYNAPSE_CLASSES = {
    1: 'aorta',
    2: 'gallbladder',
    3: 'spleen',
    4: 'left_kidney',
    5: 'right_kidney',
    6: 'liver',
    7: 'stomach',
    8: 'pancreas',
}


def _convert_to_nifti(data_dir):
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required to convert the Synapse dataset's HDF5 volumes to NIfTI. "
            'Run: pip install h5py'
        ) from e

    h5_dir = data_dir / 'Synapse' / 'test_vol_h5'
    h5_files = sorted(h5_dir.glob('*.npy.gz')) or sorted(h5_dir.glob('*.h5'))

    nii_dir = data_dir / 'nifti' / 'images'
    label_dir = data_dir / 'nifti' / 'labels'
    nii_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    label_paths = []

    for h5_path in h5_files:
        case_id = h5_path.stem.split('.')[0]  # 'case0001.npy.h5' -> 'case0001'
        nii_path = nii_dir / f'{case_id}.nii.gz'
        lbl_path = label_dir / f'{case_id}_label.nii.gz'

        if not nii_path.exists() or not lbl_path.exists():
            with h5py.File(h5_path, 'r') as f:
                image = f['image'][:]  # (H, W, D) float
                label = f['label'][:]  # (H, W, D) int

            image = image[::2, ::2, :]  # downsample in-plane
            label = label[::2, ::2, :]

            nib.save(nib.Nifti1Image(image.astype(np.float32), affine=np.eye(4)), nii_path)
            nib.save(nib.Nifti1Image(label.astype(np.uint8), affine=np.eye(4)), lbl_path)

        image_paths.append(nii_path)
        label_paths.append(lbl_path)

    return image_paths, label_paths


def create(project_name: str = _DATASET_NAME, api: Api | None = None) -> Project:
    """Download the Synapse Multi-Organ CT dataset and upload it as a Datamint project.

    Source: https://www.kaggle.com/datasets/dogcdt/synapse.
    """
    api = api or Api()

    proj, already_existed = _common.get_or_create_project(project_name, _DESCRIPTION, api)
    if already_existed:
        _LOGGER.warning(f"Project '{project_name}' already exists. Skipping data population.")
        _common.print_skip_summary(_DATASET_NAME, proj)
        return proj

    data_dir = _download.download_and_extract(_SYNAPSE_URL, 'synapse')
    image_paths, label_paths = _convert_to_nifti(data_dir)

    resource_ids = api.resources.upload_resources(
        [str(p) for p in image_paths],
        tags=['synapse', 'ct', 'abdomen'],
        publish_to=proj,
        progress_bar=True,
    )

    for lbl_path, resource_id in tqdm(zip(label_paths, resource_ids), total=len(label_paths),
                                      desc='Uploading annotations'):
        api.annotations.upload_volume_segmentation(
            resource=resource_id,
            file_path=str(lbl_path),
            name=_SYNAPSE_CLASSES,
            imported_from='Synapse Multi-Organ CT',
        )

    _common.print_summary(_DATASET_NAME, len(image_paths), len(label_paths), data_dir, proj)
    return proj
