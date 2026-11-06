import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from datamint import Api
from datamint.entities import Project

from . import _common
from ._common import ImportIssue


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
    unsupported_annotations: int
    issues: list[ImportIssue] = field(default_factory=list)
    unreferenced_images: list[str] = field(default_factory=list)

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
    """Parse a COCO-format annotations file and upload it to a Datamint project. """

    def __init__(self, annotations_file: str | Path, images_dir: str | Path | None = None):
        self.annotations_file = Path(annotations_file)
        self.images_dir = Path(images_dir) if images_dir is not None else self.annotations_file.parent
        self._result: COCOParseResult | None = None

    def parse(self, force: bool = False) -> COCOParseResult:
        """Read and validate the COCO JSON file.

        Cached after the first call; pass ``force=True`` to reparse.

        Annotations with an unknown category/image id or an invalid/out-of-bounds
        box are skipped and recorded in ``issues``. The bounds check needs the
        image ``width``/``height`` in the JSON and is skipped without them.

        Raises:
            ValueError: If the file is structurally invalid (missing required keys).
        """
        if self._result is not None and not force:
            return self._result

        with open(self.annotations_file) as f:
            data = json.load(f)

        for key in ('images', 'annotations', 'categories'):
            if key not in data:
                raise ValueError(f"Invalid COCO file: missing required key '{key}'.")

        categories = {cat['id']: cat['name'] for cat in data['categories']}
        source = self.annotations_file.name

        images_by_id: dict[int, tuple[str, Path]] = {}
        sizes_by_id: dict[int, tuple[float | None, float | None]] = {}
        missing_images: list[str] = []
        samples_by_id: dict[int, COCOSample] = {}
        for img in data['images']:
            file_name = img['file_name']
            image_path = self.images_dir / file_name
            images_by_id[img['id']] = (file_name, image_path)
            sizes_by_id[img['id']] = (img.get('width'), img.get('height'))
            if not image_path.exists():
                missing_images.append(file_name)
                continue
            samples_by_id[img['id']] = COCOSample(image_path=image_path, file_name=file_name)

        used_class_names: set[str] = set()
        unsupported_annotations = 0
        issues: list[ImportIssue] = []
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in images_by_id:
                issues.append(ImportIssue('unknown_image', source, str(image_id)))
                continue
            if image_id not in samples_by_id:
                continue  # image file missing on disk, already recorded above

            category_id = ann['category_id']
            if category_id not in categories:
                issues.append(ImportIssue('unknown_class', source, str(category_id)))
                continue

            bbox = ann.get('bbox')
            if bbox is None:
                # COCO allows annotations without bounding boxes (e.g., segmentation-only annotations)
                if ann.get('segmentation'):
                    unsupported_annotations += 1
                continue

            try:
                x, y, width, height = (float(v) for v in bbox)
            except (TypeError, ValueError):
                issues.append(ImportIssue('invalid_box', source, str(ann.get('id'))))
                continue

            issue = _common.check_box(x, y, x + width, y + height, *sizes_by_id[image_id])
            if issue is not None:
                issues.append(ImportIssue(issue, source, str(ann.get('id'))))
                continue

            label = categories[category_id]
            samples_by_id[image_id].boxes.append(COCOBox(label=label, x=x, y=y, width=width, height=height))
            used_class_names.add(label)

        referenced = {image_path for _, image_path in images_by_id.values()}
        self._result = COCOParseResult(
            samples=list(samples_by_id.values()),
            class_names=sorted(used_class_names),
            missing_images=missing_images,
            unsupported_annotations=unsupported_annotations,
            issues=issues,
            unreferenced_images=_common.find_unreferenced_images(self.images_dir, referenced, recursive=True),
        )
        return self._result

    def import_to_project(self,
                          project: Project | str,
                          api: Api | None = None,
                          *,
                          tags: Sequence[str] | None = None,
                          imported_from: str = 'coco-import',
                          on_error: Literal['raise', 'skip'] = 'raise',
                          progress_bar: bool = True) -> COCOImportResult:
        """Upload the parsed images and box annotations to a Datamint project.

        Calls :meth:`parse` first (reusing the cached result if already called).
        ``api`` defaults to a new :class:`~datamint.api.client.Api` instance if not given.
        """
        result = self.parse()
        return _common.import_boxes_to_project(
            result, project, api,
            box_points=lambda b: ((b.x, b.y), (b.x + b.width, b.y + b.height)),
            source_label=str(self.annotations_file),
            tags=tags,
            imported_from=imported_from,
            on_error=on_error,
            progress_bar=progress_bar,
            result_cls=COCOImportResult,
        )
