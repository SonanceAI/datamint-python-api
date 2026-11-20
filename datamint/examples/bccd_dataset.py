import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from datamint import Api, configs
from datamint.entities import Project
from datamint.entities.annotations import BoxAnnotation

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
        assemble_dicoms=False,
        discard_dicom_reports=False,
        progress_bar=True,
    )
    # Annotations count toward a project only through one of its worklists,
    # and annotators only see the box labels listed in its annotations.
    labels = sorted({box.label for sample in samples for box in sample.boxes})
    worklist_id = api.annotationworklists.create(configs.DEFAULT_UPLOAD_WORKLIST_NAME,
                                                 resource_ids=list(resource_ids),
                                                 project=proj,
                                                 annotations=[{'type': 'square', 'identifier': label,
                                                               'scope': 'frame', 'required': False}
                                                              for label in labels],
                                                 return_entity=False)

    annotations_to_create = [
        (
            resource_id,
            [
                BoxAnnotation.from_points(
                    (box.x1, box.y1),
                    (box.x2, box.y2),
                    identifier=box.label,
                    annotation_worklist_id=worklist_id,
                    source='imported',
                )
                for box in sample.boxes
            ],
        )
        for sample, resource_id in zip(samples, resource_ids)
        if sample.boxes
    ]
    api.annotations.bulk_create(annotations_to_create, progress_bar=True)
    n_annotated = len(annotations_to_create)

    _common.print_summary(_DATASET_NAME, len(samples), n_annotated, data_dir, proj)
    return proj
