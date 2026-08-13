import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from tqdm.auto import tqdm

from datamint import Api
from datamint.entities import Project

_LOGGER = logging.getLogger(__name__)


@dataclass
class COCOBox:
    label: str
    x: float
    y: float
    width: float
    height: float


@dataclass
class COCOSample:
    image_path: Path
    file_name: str
    boxes: list[COCOBox] = field(default_factory=list)


@dataclass
class COCOParseResult:
    samples: list[COCOSample]
    class_names: list[str]
    missing_images: list[str]

    @property
    def num_images(self) -> int:
        return len(self.samples)

    @property
    def num_boxes(self) -> int:
        return sum(len(s.boxes) for s in self.samples)


@dataclass
class COCOImportResult:
    project: Project | str
    resource_ids: list[str]
    n_images_uploaded: int
    n_boxes_uploaded: int
    errors: list[tuple[str, Exception]] = field(default_factory=list)


class COCOImporter:
    """Parse a COCO-format annotations file and upload it to a Datamint project.

    Only bounding-box annotations (the ``bbox`` field) are imported; polygon
    ``segmentation`` fields are ignored.
    """

    def __init__(self, annotations_file: str | Path, images_dir: str | Path | None = None):
        self.annotations_file = Path(annotations_file)
        self.images_dir = Path(images_dir) if images_dir is not None else self.annotations_file.parent
        self._result: COCOParseResult | None = None

    def parse(self, force: bool = False) -> COCOParseResult:
        """Read and validate the COCO JSON file.

        Cached after the first call; pass ``force=True`` to reparse.

        Raises:
            ValueError: If the file is structurally invalid (missing required
                keys, or an annotation references an unknown category/image id).
        """
        if self._result is not None and not force:
            return self._result

        with open(self.annotations_file) as f:
            data = json.load(f)

        for key in ('images', 'annotations', 'categories'):
            if key not in data:
                raise ValueError(f"Invalid COCO file: missing required key '{key}'.")

        categories = {cat['id']: cat['name'] for cat in data['categories']}

        images_by_id: dict[int, tuple[str, Path]] = {}
        missing_images: list[str] = []
        samples_by_id: dict[int, COCOSample] = {}
        for img in data['images']:
            file_name = img['file_name']
            image_path = self.images_dir / file_name
            images_by_id[img['id']] = (file_name, image_path)
            if not image_path.exists():
                missing_images.append(file_name)
                continue
            samples_by_id[img['id']] = COCOSample(image_path=image_path, file_name=file_name)

        used_class_names: set[str] = set()
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in images_by_id:
                raise ValueError(f"Annotation {ann.get('id')} references unknown image_id {image_id}.")
            if image_id not in samples_by_id:
                continue  # image file missing on disk, already recorded above

            category_id = ann['category_id']
            if category_id not in categories:
                raise ValueError(f"Annotation {ann.get('id')} references unknown category_id {category_id}.")

            bbox = ann.get('bbox')
            if bbox is None:
                # COCO allows annotations without bounding boxes (e.g., segmentation-only annotations)
                continue 

            x, y, width, height = bbox
            label = categories[category_id]
            samples_by_id[image_id].boxes.append(COCOBox(label=label, x=x, y=y, width=width, height=height))
            used_class_names.add(label)

        self._result = COCOParseResult(
            samples=list(samples_by_id.values()),
            class_names=sorted(used_class_names),
            missing_images=missing_images,
        )
        return self._result

    def import_to_project(self,
                          api: Api,
                          project: Project | str,
                          *,
                          tags: Sequence[str] | None = None,
                          imported_from: str = 'coco-import',
                          on_error: Literal['raise', 'skip'] = 'raise',
                          progress_bar: bool = True) -> COCOImportResult:
        """Upload the parsed images and box annotations to a Datamint project.

        Calls :meth:`parse` first (reusing the cached result if already called).
        """
        result = self.parse()
        if result.missing_images:
            _LOGGER.warning(f'{len(result.missing_images)} image(s) referenced in '
                            f'{self.annotations_file} were not found under {self.images_dir} and will be skipped.')

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
                        point1=(box.x, box.y),
                        point2=(box.x + box.width, box.y + box.height),
                        resource=resource_id,
                        identifier=box.label,
                        imported_from=imported_from,
                    )
                    n_boxes_uploaded += 1
                except Exception as e:
                    if on_error == 'raise':
                        raise
                    errors.append((sample.file_name, e))

        return COCOImportResult(
            project=project,
            resource_ids=resource_ids,
            n_images_uploaded=len(resource_ids),
            n_boxes_uploaded=n_boxes_uploaded,
            errors=errors,
        )
