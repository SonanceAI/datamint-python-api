import logging

from tqdm.auto import tqdm

from datamint import Api
from datamint.entities import Project

from . import _common, _download

_LOGGER = logging.getLogger(__name__)

_BUSI_URL = 'https://www.kaggle.com/api/v1/datasets/download/sabahesaraki/breast-ultrasound-images-dataset'
_DATASET_NAME = 'BUSI Segmentation Example'
_DESCRIPTION = ('Breast ultrasound segmentation example dataset (BUSI), '
               'auto-populated by datamint.examples.busi_dataset.')
_CLASSES = ('benign', 'malignant', 'normal')


def create(project_name: str = _DATASET_NAME, api: Api | None = None) -> Project:
    """Download the BUSI breast-ultrasound segmentation dataset and upload it as a Datamint project.

    Source: https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset.
    """
    api = api or Api()

    proj, already_existed = _common.get_or_create_project(project_name, _DESCRIPTION, api)
    if already_existed:
        _LOGGER.warning(f"Project '{project_name}' already exists. Skipping data population.")
        _common.print_skip_summary(_DATASET_NAME, proj)
        return proj

    data_dir = _download.download_and_extract(_BUSI_URL, 'busi')
    base_dir = data_dir / 'Dataset_BUSI_with_GT'

    image_paths = []
    mask_paths = []
    for cls in _CLASSES:
        cls_dir = base_dir / cls
        cls_images = sorted(p for p in cls_dir.glob('*.png') if '_mask' not in p.name)
        for img_path in cls_images:
            mask_path = cls_dir / f'{img_path.stem}_mask.png'
            image_paths.append(img_path)
            mask_paths.append(mask_path if mask_path.exists() else None)

    resource_ids = api.resources.upload_resources(
        [str(p) for p in image_paths],
        tags=['busi', 'ultrasound', 'breast'],
        publish_to=proj,
        progress_bar=True,
    )

    n_annotated = 0
    for img_path, mask_path, resource_id in tqdm(zip(image_paths, mask_paths, resource_ids),
                                                  total=len(image_paths),
                                                  desc='Uploading annotations'):
        if mask_path is None:
            continue  # normal images have no lesion mask
        n_annotated += 1
        api.annotations.upload_segmentations(
            resource=resource_id,
            file_path=mask_path,
            name=img_path.parent.name,  # 'benign' or 'malignant'
            imported_from='Original GT BUSI Dataset',
        )

    _common.print_summary(_DATASET_NAME, len(image_paths), n_annotated, data_dir, proj)
    return proj
