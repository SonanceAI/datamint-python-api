import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from tqdm.auto import tqdm

from datamint import Api
from datamint.entities import Project

_LOGGER = logging.getLogger(__name__)


@dataclass
class PascalVOCBox:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    difficult: bool = False


@dataclass
class PascalVOCSample:
    image_path: Path
    file_name: str
    boxes: list[PascalVOCBox] = field(default_factory=list)


@dataclass
class PascalVOCParseResult:
    samples: list[PascalVOCSample]
    class_names: list[str]
    missing_images: list[str]

    @property
    def num_images(self) -> int:
        return len(self.samples)

    @property
    def num_boxes(self) -> int:
        return sum(len(s.boxes) for s in self.samples)


@dataclass
class PascalVOCImportResult:
    project: Project | str
    resource_ids: list[str]
    n_images_uploaded: int
    n_boxes_uploaded: int
    errors: list[tuple[str, Exception]] = field(default_factory=list)


class PascalVOCImporter:
    """Parse a Pascal VOC-format annotations directory and upload it to a Datamint project.

    Only bounding-box annotations (the ``bndbox`` field) are imported.
    """

    def __init__(self, annotations_dir: str | Path, images_dir: str | Path):
        self.annotations_dir = Path(annotations_dir)
        self.images_dir = Path(images_dir)
        self._result: PascalVOCParseResult | None = None

    def parse(self, force: bool = False) -> PascalVOCParseResult:
        """Read and validate the Pascal VOC XML annotation files. 

        Cached after the first call; pass ``force=True`` to reparse.

        Raises:
            ValueError: If ``annotations_dir`` doesn't exist, or an XML file is
                missing its required ``filename`` element.
        """
        if self._result is not None and not force:
            return self._result

        if not self.annotations_dir.is_dir():
            raise ValueError(f"Invalid Pascal VOC annotations directory: '{self.annotations_dir}' does not exist.")

        samples: list[PascalVOCSample] = []
        missing_images: list[str] = []
        used_class_names: set[str] = set()

        for xml_path in sorted(self.annotations_dir.glob('*.xml')):
            root = ET.parse(xml_path).getroot()

            file_name = root.findtext('filename', default='').strip()
            if not file_name:
                raise ValueError(f"Invalid Pascal VOC annotation '{xml_path}': missing required element 'filename'.")

            image_path = self.images_dir / file_name
            if not image_path.exists():
                missing_images.append(file_name)
                continue

            sample = PascalVOCSample(image_path=image_path, file_name=file_name)
            for obj in root.findall('object'):
                label = obj.findtext('name', default='').strip()
                bb = obj.find('bndbox')
                if not label or bb is None:
                    # incomplete object entry skip it
                    continue

                difficult = obj.findtext('difficult', default='0').strip() == '1'
                sample.boxes.append(PascalVOCBox(
                    label=label,
                    x1=float(bb.findtext('xmin')),
                    y1=float(bb.findtext('ymin')),
                    x2=float(bb.findtext('xmax')),
                    y2=float(bb.findtext('ymax')),
                    difficult=difficult,
                ))
                used_class_names.add(label)

            samples.append(sample)

        self._result = PascalVOCParseResult(
            samples=samples,
            class_names=sorted(used_class_names),
            missing_images=missing_images,
        )
        return self._result

    def import_to_project(self,
                          api: Api,
                          project: Project | str,
                          *,
                          tags: Sequence[str] | None = None,
                          imported_from: str = 'pascal-voc-import',
                          on_error: Literal['raise', 'skip'] = 'raise',
                          progress_bar: bool = True) -> PascalVOCImportResult:
        """Upload the parsed images and box annotations to a Datamint project.

        Calls :meth:`parse` first (reusing the cached result if already called).
        """
        result = self.parse()
        if result.missing_images:
            _LOGGER.warning(f'{len(result.missing_images)} image(s) referenced in '
                            f'{self.annotations_dir} were not found under {self.images_dir} and will be skipped.')

        uploaded = api.resources.upload_resources(
            [str(s.image_path) for s in result.samples],
            tags=tags,
            publish_to=project,
            on_error=on_error,
            progress_bar=progress_bar,
        )

        resource_ids: list[str] = []
        errors: list[tuple[str, Exception]] = []
        n_boxes_uploaded = 0

        iterator = zip(result.samples, uploaded)
        if progress_bar:
            iterator = tqdm(iterator, total=len(result.samples), desc='Uploading annotations')

        for sample, resource_id in iterator:
            if isinstance(resource_id, Exception):
                errors.append((sample.file_name, resource_id))
                continue
            resource_ids.append(resource_id)

            for box in sample.boxes:
                try:
                    api.annotations.add_box_annotation(
                        point1=(box.x1, box.y1),
                        point2=(box.x2, box.y2),
                        resource=resource_id,
                        identifier=box.label,
                        imported_from=imported_from,
                    )
                    n_boxes_uploaded += 1
                except Exception as e:
                    if on_error == 'raise':
                        raise
                    errors.append((sample.file_name, e))

        return PascalVOCImportResult(
            project=project,
            resource_ids=resource_ids,
            n_images_uploaded=len(resource_ids),
            n_boxes_uploaded=n_boxes_uploaded,
            errors=errors,
        )
