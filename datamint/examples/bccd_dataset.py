import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from tqdm.auto import tqdm

from datamint import Api
from datamint.entities import Project

from . import _common, _download

_LOGGER = logging.getLogger(__name__)

_BCCD_URL = 'https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip'
_DATASET_NAME = 'BCCD Detection Example'
_DESCRIPTION = ('Blood cell detection example dataset (BCCD), '
               'auto-populated by datamint.examples.bccd_dataset.')


@dataclass
class _Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class _Sample:
    image_path: Path
    boxes: list[_Box] = field(default_factory=list)


def _parse_voc_xml(xml_path: Path) -> list[_Box]:
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall('object'):
        name = obj.findtext('name', default='').strip()
        bb = obj.find('bndbox')
        if bb is None:
            continue
        boxes.append(_Box(
            label=name,
            x1=float(bb.findtext('xmin')),
            y1=float(bb.findtext('ymin')),
            x2=float(bb.findtext('xmax')),
            y2=float(bb.findtext('ymax')),
        ))
    return boxes


def create(project_name: str = _DATASET_NAME, api: Api | None = None) -> Project:
    """Download the BCCD blood-cell detection dataset and upload it as a Datamint project.

    Source: https://github.com/Shenggan/BCCD_Dataset (MIT License).
    """
    api = api or Api()

    proj, already_existed = _common.get_or_create_project(project_name, _DESCRIPTION, api)
    if already_existed:
        _LOGGER.warning(f"Project '{project_name}' already exists. Skipping data population.")
        _common.print_skip_summary(_DATASET_NAME, proj)
        return proj

    data_dir = _download.download_and_extract(_BCCD_URL, 'bccd')
    images_dir = data_dir / 'BCCD_Dataset-master' / 'BCCD' / 'JPEGImages'
    annots_dir = data_dir / 'BCCD_Dataset-master' / 'BCCD' / 'Annotations'

    image_files = sorted(images_dir.glob('*.jpg'))
    samples = []
    for img_path in image_files:
        xml_path = annots_dir / img_path.with_suffix('.xml').name
        samples.append(_Sample(
            image_path=img_path,
            boxes=_parse_voc_xml(xml_path) if xml_path.exists() else [],
        ))

    resource_ids = api.resources.upload_resources(
        [str(s.image_path) for s in samples],
        tags=['bccd'],
        publish_to=proj,
        progress_bar=True,
    )

    n_annotated = 0
    for sample, resource_id in tqdm(zip(samples, resource_ids), total=len(samples),
                                    desc='Uploading annotations'):
        if not sample.boxes:
            continue
        n_annotated += 1
        for box in sample.boxes:
            api.annotations.add_box_annotation(
                point1=(box.x1, box.y1),
                point2=(box.x2, box.y2),
                resource=resource_id,
                identifier=box.label,
            )

    _common.print_summary(_DATASET_NAME, len(samples), n_annotated, data_dir, proj)
    return proj
