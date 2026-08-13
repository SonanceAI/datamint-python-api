from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import yaml
from PIL import Image

from datamint import Api
from datamint.entities import Project

from . import _common

_DEFAULT_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


@dataclass
class YOLOBox:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class YOLOSample:
    image_path: Path
    file_name: str
    boxes: list[YOLOBox] = field(default_factory=list)


@dataclass
class YOLOParseResult:
    samples: list[YOLOSample]
    class_names: list[str]
    missing_images: list[str]
    unsupported_annotations: int

    @property
    def num_images(self) -> int:
        return len(self.samples)

    @property
    def num_boxes(self) -> int:
        return sum(len(s.boxes) for s in self.samples)


@dataclass
class YOLOImportResult:
    project: Project | str
    resource_ids: list[str]
    n_images_uploaded: int
    n_boxes_uploaded: int
    errors: list[tuple[str, Exception]] = field(default_factory=list)


def _load_names_from_yaml(path: Path) -> dict[int, str]:
    with open(path) as f:
        data = yaml.safe_load(f)

    names = data.get('names')
    if names is None:
        raise ValueError(f"Invalid YOLO data.yaml '{path}': missing required key 'names'.")

    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    return {idx: str(name) for idx, name in enumerate(names)}


class YOLOImporter:
    """Parse a YOLO-format (images + normalized-bbox label .txt files) dataset and
    upload it to a Datamint project.

    """

    def __init__(self,
                images_dir: str | Path,
                labels_dir: str | Path,
                *,
                class_names: Sequence[str] | None = None,
                data_yaml: str | Path | None = None,
                image_extensions: Sequence[str] = _DEFAULT_IMAGE_EXTENSIONS):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names_override = list(class_names) if class_names is not None else None
        self.data_yaml = Path(data_yaml) if data_yaml is not None else None
        self.image_extensions = image_extensions
        self._result: YOLOParseResult | None = None

    def _resolve_class_names(self) -> dict[int, str]:
        if self.class_names_override is not None:
            return dict(enumerate(self.class_names_override))

        if self.data_yaml is not None:
            return _load_names_from_yaml(self.data_yaml)

        raise ValueError('No class names available: pass class_names or data_yaml explicitly.')

    def _find_image(self, stem: str) -> Path | None:
        for ext in self.image_extensions:
            candidate = self.images_dir / f'{stem}{ext}'
            if candidate.exists():
                return candidate
        return None

    def parse(self, force: bool = False) -> YOLOParseResult:
        """Read and validate the YOLO label files.

        Cached after the first call; pass ``force=True`` to reparse.

        Raises:
            ValueError: If class names can't be resolved, a label line
                references an unknown class index, or a normalized coordinate
                is outside the expected ``[0, 1]`` range.
        """
        if self._result is not None and not force:
            return self._result

        class_names_by_id = self._resolve_class_names()

        samples: list[YOLOSample] = []
        missing_images: list[str] = []
        used_class_names: set[str] = set()
        unsupported_annotations = 0

        for txt_path in sorted(self.labels_dir.glob('*.txt')):
            if txt_path.name in ('classes.txt',):
                continue

            image_path = self._find_image(txt_path.stem)
            if image_path is None:
                missing_images.append(txt_path.name)
                continue

            img_w, img_h = Image.open(image_path).size

            sample = YOLOSample(image_path=image_path, file_name=image_path.name)
            with open(txt_path) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        # segmentation/OBB/keypoint variants have a different field count -- not supported
                        unsupported_annotations += 1
                        continue

                    class_id = int(parts[0])
                    if class_id not in class_names_by_id:
                        raise ValueError(f"Label '{txt_path}' references unknown class_id {class_id}.")

                    cx, cy, w, h = (float(v) for v in parts[1:])
                    for coord_name, value in (('x_center', cx), ('y_center', cy), ('width', w), ('height', h)):
                        if not (0.0 <= value <= 1.0):
                            raise ValueError(f"Label '{txt_path}' has out-of-range normalized "
                                            f"{coord_name}={value!r} (expected 0<=v<=1).")

                    cx, w = cx * img_w, w * img_w
                    cy, h = cy * img_h, h * img_h

                    label = class_names_by_id[class_id]
                    sample.boxes.append(YOLOBox(
                        label=label,
                        x1=cx - w / 2,
                        y1=cy - h / 2,
                        x2=cx + w / 2,
                        y2=cy + h / 2,
                    ))
                    used_class_names.add(label)

            samples.append(sample)

        self._result = YOLOParseResult(
            samples=samples,
            class_names=sorted(used_class_names),
            missing_images=missing_images,
            unsupported_annotations=unsupported_annotations,
        )
        return self._result

    def import_to_project(self,
                          project: Project | str,
                          api: Api | None = None,
                          *,
                          tags: Sequence[str] | None = None,
                          imported_from: str = 'yolo-import',
                          on_error: Literal['raise', 'skip'] = 'raise',
                          progress_bar: bool = True) -> YOLOImportResult:
        """Upload the parsed images and box annotations to a Datamint project.

        Calls :meth:`parse` first (reusing the cached result if already called).
        ``api`` defaults to a new :class:`~datamint.api.client.Api` instance if not given.
        """
        result = self.parse()
        return _common.import_boxes_to_project(
            result, project, api,
            box_points=lambda b: ((b.x1, b.y1), (b.x2, b.y2)),
            source_label=str(self.labels_dir),
            tags=tags,
            imported_from=imported_from,
            on_error=on_error,
            progress_bar=progress_bar,
            result_cls=YOLOImportResult,
        )
