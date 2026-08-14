import logging
from typing import Callable, Literal, Sequence

from tqdm.auto import tqdm

from datamint import Api
from datamint.entities import Project

_LOGGER = logging.getLogger(__name__)


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
                            result_cls: type):
    """Shared upload loop behind every ``*Importer.import_to_project()``.

    ``result`` is a parse result with ``samples``, ``missing_images``, and
    ``unsupported_annotations`` attributes. ``box_points`` maps a format's box
    dataclass to a ``(point1, point2)`` pair for ``add_box_annotation``.

    """
    api = api or Api()

    if result.missing_images:
        _LOGGER.warning(f'{len(result.missing_images)} image(s) referenced in {source_label} '
                        f'were not found on disk and will be skipped.')
    if result.unsupported_annotations:
        _LOGGER.warning(f'{result.unsupported_annotations} annotation(s) in {source_label} use an '
                        f'unsupported annotation type (e.g. polygon/segmentation) and were not '
                        f'imported; only bounding boxes are supported.')

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
