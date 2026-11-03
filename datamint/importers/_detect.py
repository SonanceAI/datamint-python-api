"""Best-effort detection of a labeled dataset's annotation format.

Looks at what's on disk under a root directory and guesses which
``*Importer`` in :mod:`datamint.importers` applies, along with the concrete
paths that importer's constructor needs. Ambiguous or non-standard layouts should
fall back to an explicit format choice + path overrides rather than a
silent wrong guess.
"""
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DatasetFormat = Literal['coco', 'yolo', 'pascal_voc']

_IMAGE_DIR_NAMES = {'images', 'image', 'imgs', 'jpegimages', 'img'}
_LABEL_DIR_NAMES = {'labels', 'label'}


@dataclass
class DetectedDataset:
    """Result of a format sniff: which format, and the constructor kwargs
    to feed the matching ``*Importer``."""
    format: DatasetFormat
    importer_kwargs: dict[str, Path | list[Path]] = field(default_factory=dict)


def _find_dir(root: Path, names: set[str]) -> Path | None:
    for p in sorted(root.rglob('*')):
        if p.is_dir() and p.name.lower() in names:
            return p
    return None


def sniff_coco(root: Path) -> DetectedDataset | None:
    """Look for a JSON file with the COCO ``images``/``annotations``/``categories`` keys."""
    for json_path in sorted(root.rglob('*.json')):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and {'images', 'annotations', 'categories'} <= data.keys():
            images_dir = _find_dir(root, _IMAGE_DIR_NAMES) or json_path.parent
            return DetectedDataset('coco', {'annotations_file': json_path, 'images_dir': images_dir})
    return None


def sniff_pascal_voc(root: Path) -> DetectedDataset | None:
    """Look for ``.xml`` files whose root element is a Pascal VOC ``<annotation>``."""
    xml_paths = sorted(root.rglob('*.xml'))
    for xml_path in xml_paths[:5]:  
        try:
            root_el = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        if root_el.tag == 'annotation' and root_el.find('object') is not None:
            annotations_dir = xml_path.parent
            images_dir = _find_dir(root, _IMAGE_DIR_NAMES) or annotations_dir
            return DetectedDataset('pascal_voc', {'annotations_dir': annotations_dir, 'images_dir': images_dir})
    return None


def _is_yolo_labels_dir(path: Path) -> bool:
    txt_paths = [p for p in sorted(path.glob('*.txt')) if p.name != 'classes.txt']
    for txt_path in txt_paths[:5]:
        values = txt_path.read_text().split()
        # YOLO box line: class cx cy w h
        if values and len(values) % 5 == 0:
            return True
    return False


def _yolo_images_dir(labels_dir: Path) -> Path:
    """Swap the last ``labels`` component of the path for its sibling images dir
    (``train/labels`` -> ``train/images``, ``labels/train`` -> ``images/train``)."""
    parts = labels_dir.parts
    for i in reversed(range(len(parts))):
        if parts[i].lower() not in _LABEL_DIR_NAMES:
            continue
        for sibling in sorted(Path(*parts[:i]).iterdir()):
            candidate = sibling.joinpath(*parts[i + 1:])
            if sibling.is_dir() and sibling.name.lower() in _IMAGE_DIR_NAMES and candidate.is_dir():
                return candidate
        break
    return labels_dir.parent / 'images'


def sniff_yolo(root: Path) -> DetectedDataset | None:
    """Look for ``labels/`` dirs of YOLO ``.txt`` files (class + 4 normalized floats).

    Every split (``train``, ``valid``, ``test``, ...) is picked up, each paired with its images dir.
    """
    labels_dirs = [p for p in sorted(root.rglob('*'))
                   if p.is_dir()
                   and any(part.lower() in _LABEL_DIR_NAMES for part in p.relative_to(root).parts)
                   and _is_yolo_labels_dir(p)]
    if not labels_dirs:
        return None

    images_dirs = [_yolo_images_dir(p) for p in labels_dirs]
    kwargs: dict[str, Path | list[Path]] = {
        'images_dir': images_dirs[0] if len(images_dirs) == 1 else images_dirs,
        'labels_dir': labels_dirs[0] if len(labels_dirs) == 1 else labels_dirs,
    }

    data_yaml = next(root.glob('*.yaml'), None) or next(root.glob('*.yml'), None)
    if data_yaml is not None:
        kwargs['data_yaml'] = data_yaml
    return DetectedDataset('yolo', kwargs)


_SNIFFERS = {
    'coco': sniff_coco,
    'pascal_voc': sniff_pascal_voc,
    'yolo': sniff_yolo,
}


def detect_format(root: str | Path) -> DetectedDataset | None:
    """Guess the annotation format of a dataset directory.

    Tries COCO, then Pascal VOC, then YOLO (see the corresponding
    ``sniff_*`` function for what triggers each). Returns ``None`` if
    nothing matched.
    """
    root = Path(root)
    for sniff in _SNIFFERS.values():
        result = sniff(root)
        if result is not None:
            return result
    return None


def sniff_single_format(root: str | Path, format: DatasetFormat) -> DetectedDataset | None:
    """Locate paths for a single, already-chosen format."""
    return _SNIFFERS[format](Path(root))
