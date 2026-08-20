"""datamint import command-line tool.

Uploads an already-labeled dataset (COCO, Pascal VOC, or YOLO format) to a
Datamint project in one call -- images plus their bounding-box annotations.
See ``datamint import --help`` for the available options.
"""
import argparse
import logging
import sys
from pathlib import Path

from datamint.client_cmd_tools.datamint_upload import handle_api_key
from datamint.exceptions import DatamintException
from datamint.importers import (
    COCOImporter,
    DatasetFormat,
    PascalVOCImporter,
    YOLOImporter,
    detect_format,
    sniff_single_format,
)
from datamint.utils.logging_utils import load_cmdline_logging_config

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')

_IMPORTER_CLASSES = {
    'coco': COCOImporter,
    'pascal_voc': PascalVOCImporter,
    'yolo': YOLOImporter,
}

# Which CLI override flags apply to each format's importer constructor.
_FORMAT_OVERRIDE_KEYS = {
    'coco': ('annotations_file', 'images_dir'),
    'pascal_voc': ('annotations_dir', 'images_dir'),
    'yolo': ('images_dir', 'labels_dir', 'class_names', 'data_yaml'),
}

_REQUIRED_KEYS = {
    'coco': ('annotations_file', 'images_dir'),
    'pascal_voc': ('annotations_dir', 'images_dir'),
    'yolo': ('images_dir', 'labels_dir'),
}


def _build_parser(subparsers: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build the argument parser.

    When ``subparsers`` is given, the parser is registered as an ``import`` subparser
    (used by ``datamint``'s combined completion tree) instead of a standalone parser.
    """
    kwargs = {
        'description': 'Upload an already-labeled dataset (COCO, Pascal VOC, or YOLO) to a Datamint project.',
        'epilog': """
Examples:
  datamint import ./my_coco_dataset --project MyProject          # auto-detect the format
  datamint import ./my_yolo_dataset --project MyProject --format yolo
  datamint import ./voc_dataset --project MyProject --dry-run    # preview without uploading

More Documentation: https://sonanceai.github.io/datamint-python-api/command_line_tools.html
        """,
        'formatter_class': argparse.RawDescriptionHelpFormatter,
    }
    if subparsers is not None:
        parser = subparsers.add_parser('import', **kwargs)
    else:
        parser = argparse.ArgumentParser(**kwargs)

    parser.add_argument('path', type=str, metavar='PATH', help='Root directory of the labeled dataset.')
    parser.add_argument('--project', type=str, required=True,
                        help='Name of the Datamint project to import into. Created if it does not exist yet.')
    parser.add_argument('--format', type=str, choices=sorted(_IMPORTER_CLASSES),
                        help='Annotation format. If omitted, the format is auto-detected from PATH.')

    detect_group = parser.add_argument_group(
        'Path overrides',
        'Override the paths that auto-detection would otherwise infer. Required when --format '
        'is given and the layout could not be located automatically.',
    )
    detect_group.add_argument('--annotations-file', type=str, help='(COCO) Path to the annotations JSON file.')
    detect_group.add_argument('--annotations-dir', type=str, help='(Pascal VOC) Directory of annotation XML files.')
    detect_group.add_argument('--labels-dir', type=str, help='(YOLO) Directory of label .txt files.')
    detect_group.add_argument('--images-dir', type=str, help='Directory of image files.')
    detect_group.add_argument('--class-names', type=str, nargs='+',
                              help='(YOLO) Ordered class names, index 0 first. Alternative to --data-yaml.')
    detect_group.add_argument('--data-yaml', type=str, help='(YOLO) data.yaml file providing class names.')

    parser.add_argument('--tag', type=str, action='append', help='A tag to apply to every uploaded resource.')
    parser.add_argument('--imported-from', type=str, default=None,
                        help="Provenance label stored on each annotation. Defaults to '<format>-import'.")
    parser.add_argument('--on-error', type=str, choices=('raise', 'skip'), default='raise',
                        help='Whether to stop or skip past a single resource/annotation upload failure.')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Parse and print a summary, but do not create the project or upload anything.')
    parser.add_argument('--yes', action='store_true', default=False,
                        help='Skip the auto-detected-format confirmation prompt.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print debug messages.')

    return parser


def _parse_args() -> argparse.Namespace:
    parser = _build_parser()
    import argcomplete
    argcomplete.autocomplete(parser)
    return parser.parse_args()


def _resolve_dataset(args: argparse.Namespace) -> tuple[DatasetFormat, dict]:
    """Figure out the format + importer constructor kwargs, applying CLI overrides.

    Prompts for confirmation when the format was auto-detected (unless ``--yes``).
    Raises ``DatamintException`` when the format/paths can't be resolved.
    """
    path = Path(args.path)

    if args.format is not None:
        fmt: DatasetFormat = args.format
        detected = sniff_single_format(path, fmt)
        base_kwargs = dict(detected.importer_kwargs) if detected is not None else {}
    else:
        detected = detect_format(path)
        if detected is None:
            raise DatamintException(
                f"Could not auto-detect a dataset format under '{path}'. "
                f"Pass --format explicitly ({', '.join(sorted(_IMPORTER_CLASSES))}) "
                'together with the matching path override flags.'
            )
        fmt = detected.format
        base_kwargs = dict(detected.importer_kwargs)

        _USER_LOGGER.info(f"Detected a {fmt.upper()} dataset under '{path}':")
        for key, value in base_kwargs.items():
            _USER_LOGGER.info(f'  {key}: {value}')
        if not args.yes:
            answer = input(f'Import as {fmt.upper()}? (y/n): ')
            if answer.strip().lower() != 'y':
                raise DatamintException(
                    'Aborted. Re-run with --format to choose a different format explicitly.'
                )

    overrides = {
        'annotations_file': args.annotations_file,
        'annotations_dir': args.annotations_dir,
        'labels_dir': args.labels_dir,
        'images_dir': args.images_dir,
        'class_names': args.class_names,
        'data_yaml': args.data_yaml,
    }
    for key in _FORMAT_OVERRIDE_KEYS[fmt]:
        value = overrides.get(key)
        if value is not None:
            base_kwargs[key] = Path(value) if key != 'class_names' else value

    missing = [key for key in _REQUIRED_KEYS[fmt] if key not in base_kwargs]
    if missing:
        flags = ', '.join(f'--{key.replace("_", "-")}' for key in missing)
        raise DatamintException(
            f"Could not locate the {fmt.upper()} dataset layout under '{path}'. "
            f'Pass {flags} explicitly.'
        )

    return fmt, base_kwargs


def main() -> None:
    load_cmdline_logging_config()

    args = _parse_args()

    if args.verbose:
        logging.getLogger().handlers[0].setLevel(logging.DEBUG)
        logging.getLogger('datamint').setLevel(logging.DEBUG)
        _LOGGER.setLevel(logging.DEBUG)
        _USER_LOGGER.setLevel(logging.DEBUG)

    try:
        path = Path(args.path)
        if not path.is_dir():
            raise DatamintException(f"'{path}' is not a directory.")

        fmt, importer_kwargs = _resolve_dataset(args)
        importer = _IMPORTER_CLASSES[fmt](**importer_kwargs)

        result = importer.parse()
        _USER_LOGGER.info(
            f'Parsed {result.num_images} image(s), {result.num_boxes} box(es), '
            f'{len(result.class_names)} class(es): {result.class_names}'
        )
        if result.missing_images:
            _USER_LOGGER.warning(f'{len(result.missing_images)} referenced image(s) not found on disk.')
        if result.unsupported_annotations:
            _USER_LOGGER.warning(
                f'{result.unsupported_annotations} annotation(s) use an unsupported type and will be skipped.'
            )

        if args.dry_run:
            _USER_LOGGER.info('Dry run: nothing was uploaded.')
            return

        api_key = handle_api_key()
        if api_key is None:
            _USER_LOGGER.error('API key not provided. Aborting.')
            sys.exit(1)
        import os

        from datamint import Api, configs
        os.environ[configs.ENV_VARS[configs.APIKEY_KEY]] = api_key

        api = Api(check_connection=True)
        project = api.projects.create(
            name=args.project,
            description=f"Imported via 'datamint import' ({fmt} format).",
            exists_ok=True,
        )

        import_kwargs = {}
        if args.imported_from is not None:
            import_kwargs['imported_from'] = args.imported_from

        import_result = importer.import_to_project(
            project=project,
            api=api,
            tags=args.tag,
            on_error=args.on_error,
            progress_bar=True,
            **import_kwargs,
        )

        _USER_LOGGER.info(
            f'Uploaded {import_result.n_images_uploaded} image(s) and '
            f'{import_result.n_boxes_uploaded} box annotation(s) to project {args.project!r}.'
        )
        if import_result.errors:
            _USER_LOGGER.warning(f'{len(import_result.errors)} error(s) during upload:')
            for file_name, error in import_result.errors:
                _USER_LOGGER.warning(f'  {file_name}: {error}')
        _USER_LOGGER.info(project.url)
    except DatamintException as e:
        _USER_LOGGER.error(f'❌ {e}')
        sys.exit(1)
    except KeyboardInterrupt:
        _USER_LOGGER.warning('\nCancelled by user.')
        sys.exit(1)


if __name__ == '__main__':
    main()
