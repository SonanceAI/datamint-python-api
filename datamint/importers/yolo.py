from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import yaml
from PIL import Image

from datamint import Api
from datamint.entities import Project

from . import _common
from ._common import ImportIssue
from ._detect import _LABEL_DIR_NAMES

_DEFAULT_IMAGE_EXTENSIONS = _common.IMAGE_EXTENSIONS


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
    split: str | None = None


@dataclass
class YOLOParseResult:
    samples: list[YOLOSample]
    class_names: list[str]
    missing_images: list[str]
    unsupported_annotations: int
    
    # list of issues and unreferenced images are not used for the import, but can be useful for debugging
    issues: list[ImportIssue] = field(default_factory=list)
    unreferenced_images: list[str] = field(default_factory=list)

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


def _as_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(v) for v in value]


def _split_name(labels_dir: Path) -> str:
    """``train/labels`` -> ``train``, ``labels/train`` -> ``train``."""
    parts = labels_dir.parts
    idxs = [i for i, part in enumerate(parts) if part.lower() in _LABEL_DIR_NAMES]
    if not idxs:
        return labels_dir.name
    idx = idxs[-1]
    return parts[idx + 1] if idx + 1 < len(parts) else parts[idx - 1]


class YOLOImporter:
    """Parse a YOLO-format (images + normalized-bbox label .txt files) dataset and
    upload it to a Datamint project.

    ``images_dir`` and ``labels_dir`` take one dir each, or one per split, paired by position.
    With several splits, each resource is tagged ``split:<name>`` (e.g. ``split:train``).
    """

    def __init__(self,
                images_dir: str | Path | Sequence[str | Path],
                labels_dir: str | Path | Sequence[str | Path],
                *,
                class_names: Sequence[str] | None = None,
                data_yaml: str | Path | None = None,
                image_extensions: Sequence[str] = _DEFAULT_IMAGE_EXTENSIONS):
        images_dirs = _as_paths(images_dir)
        labels_dirs = _as_paths(labels_dir)
        if len(images_dirs) != len(labels_dirs):
            raise ValueError(f'images_dir and labels_dir must have the same number of entries '
                             f'(got {len(images_dirs)} and {len(labels_dirs)}).')
        self.splits = list(zip(images_dirs, labels_dirs))
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

    def _find_image(self, images_dir: Path, stem: str) -> Path | None:
        for ext in self.image_extensions:
            candidate = images_dir / f'{stem}{ext}'
            if candidate.exists():
                return candidate
        return None

    def parse(self, force: bool = False) -> YOLOParseResult:
        """Read and validate the YOLO label files.

        Cached after the first call; pass ``force=True`` to reparse.

        Lines with an unknown class index, unparsable values, or a box outside
        the image are skipped and recorded in ``issues``.

        Raises:
            ValueError: If class names can't be resolved.
        """
        if self._result is not None and not force:
            return self._result

        class_names_by_id = self._resolve_class_names()

        samples: list[YOLOSample] = []
        missing_images: list[str] = []
        used_class_names: set[str] = set()
        unsupported_annotations = 0
        issues: list[ImportIssue] = []
        unreferenced_images: list[str] = []

        tag_splits = len(self.splits) > 1
        for images_dir, labels_dir in self.splits:
            split = _split_name(labels_dir) if tag_splits else None
            referenced: set[Path] = set()
            for txt_path in sorted(labels_dir.glob('*.txt')):
                if txt_path.name in ('classes.txt',):
                    continue

                image_path = self._find_image(images_dir, txt_path.stem)
                if image_path is None:
                    missing_images.append(txt_path.name)
                    continue
                referenced.add(image_path)

                img_w, img_h = Image.open(image_path).size

                sample = YOLOSample(image_path=image_path, file_name=image_path.name, split=split)
                with open(txt_path) as f:
                    for line in f:
                        parts = line.split()
                        if not parts:
                            continue
                        if len(parts) != 5:
                            # segmentation/OBB/keypoint variants have a different field count -- not supported
                            unsupported_annotations += 1
                            continue

                        try:
                            class_id = int(parts[0])
                            cx, cy, w, h = (float(v) for v in parts[1:])
                        except ValueError:
                            issues.append(ImportIssue('invalid_box', txt_path.name, line.strip()))
                            continue

                        if class_id not in class_names_by_id:
                            issues.append(ImportIssue('unknown_class', txt_path.name, str(class_id)))
                            continue

                        cx, w = cx * img_w, w * img_w
                        cy, h = cy * img_h, h * img_h
                        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

                        issue = _common.check_box(x1, y1, x2, y2, img_w, img_h)
                        if issue is not None:
                            issues.append(ImportIssue(issue, txt_path.name))
                            continue

                        label = class_names_by_id[class_id]
                        sample.boxes.append(YOLOBox(label=label, x1=x1, y1=y1, x2=x2, y2=y2))
                        used_class_names.add(label)

                samples.append(sample)

            unreferenced_images += _common.find_unreferenced_images(images_dir, referenced)

        self._result = YOLOParseResult(
            samples=samples,
            class_names=sorted(used_class_names),
            missing_images=missing_images,
            unsupported_annotations=unsupported_annotations,
            issues=issues,
            unreferenced_images=unreferenced_images,
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
            source_label=', '.join(str(labels_dir) for _, labels_dir in self.splits),
            sample_tags=lambda s: [f'split:{s.split}'] if s.split else [],
            tags=tags,
            imported_from=imported_from,
            on_error=on_error,
            progress_bar=progress_bar,
            result_cls=YOLOImportResult,
        )
