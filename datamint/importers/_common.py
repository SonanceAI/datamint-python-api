import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

from tqdm.auto import tqdm

from datamint import Api
from datamint.entities import Project

_LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

IssueKind = Literal['unknown_class', 'out_of_bounds', 'invalid_box', 'unknown_image', 'invalid_file']


@dataclass
class ImportIssue:
    """A problem found while parsing. The affected box, annotation or file is skipped."""
    kind: IssueKind
    source: str
    detail: str = ''


def check_box(x1: float, y1: float, x2: float, y2: float,
              img_w: float | None, img_h: float | None) -> IssueKind | None:
    """Return the issue kind for a pixel-space box, or None if it is valid.

    The bounds check is skipped when the image size is unknown.
    """
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return 'invalid_box'
    if not img_w or not img_h:
        return None
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return 'out_of_bounds'
    return None


def find_unreferenced_images(images_dir: Path, referenced: set[Path], recursive: bool = False) -> list[str]:
    """Image files under ``images_dir`` that no annotation refers to (they are not uploaded)."""
    if not images_dir.is_dir():
        return []
    referenced = {p.resolve() for p in referenced}
    paths = images_dir.rglob('*') if recursive else images_dir.iterdir()
    return sorted(str(p.relative_to(images_dir)) for p in paths
                  if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file() and p.resolve() not in referenced)


# --- Report ---

_FORMAT_NAMES = {'coco': 'COCO', 'pascal_voc': 'Pascal VOC', 'yolo': 'YOLO'}

_ISSUE_MESSAGES = {
    'unknown_class': 'box(es) with unknown class',
    'out_of_bounds': 'box(es) outside image bounds',
    'invalid_box': 'invalid box(es)',
    'unknown_image': 'annotation(s) referencing unknown image ids',
    'invalid_file': 'unreadable annotation file(s)',
}


def summarize_issues(result) -> list[str]:
    """One line per kind of problem found while parsing."""
    lines = []
    if result.missing_images:
        lines.append(f'{len(result.missing_images)} referenced image(s) not found on disk')
    if result.unsupported_annotations:
        lines.append(f'{result.unsupported_annotations} unsupported annotation(s), only boxes are imported')
    for kind, n in Counter(issue.kind for issue in result.issues).items():
        line = f'{n} {_ISSUE_MESSAGES[kind]}'
        if kind == 'unknown_class':
            line += f' {sorted({issue.detail for issue in result.issues if issue.kind == kind})}'
        lines.append(line)
    return lines


def build_report(fmt: str, result, *, detected: bool, paths: dict, project: str | None,
                 dry_run: bool) -> list[tuple[str, str]]:
    """(key, value) rows describing what the import found and what it will do."""
    n_empty = sum(1 for s in result.samples if not s.boxes)
    splits = Counter(s.split for s in result.samples if getattr(s, 'split', None))
    issues = summarize_issues(result)
    verb = 'would upload' if dry_run else 'will upload'
    target = f"to project '{project}' (created if missing)" if project else '(no --project given)'

    rows = [('format', f"{_FORMAT_NAMES[fmt]} ({'detected' if detected else 'given'})")]
    for key, value in paths.items():
        if key != 'class_names':
            rows.append((key, ', '.join(map(str, value)) if isinstance(value, list) else str(value)))
    rows += [
        ('images', f'{result.num_images:,} ({len(result.unreferenced_images):,} on disk not referenced, skipped)'),
        ('labeled', f'{result.num_images - n_empty:,} ({n_empty:,} without boxes)'),
    ]
    if splits:
        rows.append(('splits', ' · '.join(f'{name} {n:,}' for name, n in sorted(splits.items()))))
    rows += [
        ('boxes', f'{result.num_boxes:,}'),
        ('classes', f'{len(result.class_names)} · {result.class_names}'),
        ('issues', ' · '.join(issues) if issues else 'none'),
        ('action', f'{verb} {result.num_images:,} image(s) and {result.num_boxes:,} box(es) {target}'),
    ]
    return rows


def format_report(rows: list[tuple[str, str]]) -> str:
    width = max(len(key) for key, _ in rows) + 2
    return '\n'.join(f'{key:<{width}}{value}' for key, value in rows)


def import_boxes_to_project(result,
                            project: Project | str,
                            api: Api | None,
                            *,
                            box_points: Callable[[object], tuple[tuple[float, float], tuple[float, float]]],
                            source_label: str,
                            tags: Sequence[str] | None,
                            imported_from: str,
                            on_error: Literal['raise', 'skip'],
                            progress_bar: bool,
                            result_cls: type,
                            sample_tags: Callable[[object], Sequence[str]] | None = None):
    """Shared upload loop behind every ``*Importer.import_to_project()``.

    ``result`` is a parse result with ``samples``, ``missing_images``, and
    ``unsupported_annotations`` attributes. ``box_points`` maps a format's box
    dataclass to a ``(point1, point2)`` pair for ``add_box_annotation``.
    ``sample_tags`` gives extra tags per sample (e.g. its split), added to ``tags``.

    """
    api = api or Api()

    if result.missing_images:
        _LOGGER.warning(f'{len(result.missing_images)} image(s) referenced in {source_label} '
                        f'were not found on disk and will be skipped.')
    if result.unsupported_annotations:
        _LOGGER.warning(f'{result.unsupported_annotations} annotation(s) in {source_label} use an '
                        f'unsupported annotation type (e.g. polygon/segmentation) and were not '
                        f'imported; only bounding boxes are supported.')
    if result.issues:
        _LOGGER.warning(f'{len(result.issues)} box(es)/annotation(s) in {source_label} have issues '
                        f'(unknown class, invalid or out-of-bounds box, ...) and were not imported.')

    # upload_resources applies one tag list to all files, so samples are uploaded in groups of equal tags
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, sample in enumerate(result.samples):
        extra = tuple(sample_tags(sample)) if sample_tags is not None else ()
        groups.setdefault(extra, []).append(i)

    uploaded: list = [None] * len(result.samples)
    for extra, idxs in groups.items():
        group_tags = list(tags or []) + list(extra)
        group_ids = api.resources.upload_resources(
            [str(result.samples[i].image_path) for i in idxs],
            tags=group_tags or None,
            publish_to=project,
            on_error=on_error,
            progress_bar=progress_bar,
        )
        for i, resource_id in zip(idxs, group_ids):
            uploaded[i] = resource_id

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
            point1, point2 = box_points(box)
            try:
                api.annotations.add_box_annotation(
                    point1=point1,
                    point2=point2,
                    resource=resource_id,
                    identifier=box.label,
                    imported_from=imported_from,
                )
                n_boxes_uploaded += 1
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                if on_error == 'raise':
                    raise
                errors.append((sample.file_name, e))

    return result_cls(
        project=project,
        resource_ids=resource_ids,
        n_images_uploaded=len(resource_ids),
        n_boxes_uploaded=n_boxes_uploaded,
        errors=errors,
    )
