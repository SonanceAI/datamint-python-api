"""datamint download command-line tool.

Downloads and caches a Datamint project's resources locally, then prints the
one-liner needed to load them as a PyTorch dataset with the SDK.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from datamint.client_cmd_tools.datamint_upload import handle_api_key
from datamint.exceptions import DatamintException
from datamint.utils.logging_utils import load_cmdline_logging_config

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')


def _build_parser(subparsers: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build the argument parser."""
    
    kwargs = {
        'description': "Download and cache a Datamint project's dataset locally.",
        'epilog': """
Examples:
  datamint download --project MyProject                     # cache the dataset locally
  datamint download --project MyProject -o ./my_data_folder  # also create a browsable view

More Documentation: https://sonanceai.github.io/datamint-python-api/command_line_tools.html
        """,
        'formatter_class': argparse.RawDescriptionHelpFormatter,
    }
    if subparsers is not None:
        parser = subparsers.add_parser('download', **kwargs)
    else:
        parser = argparse.ArgumentParser(**kwargs)
    parser.add_argument('--project', type=str, required=True,
                        help='Name of the Datamint project to download.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Optional folder for a browsable view of the downloaded files '
                             '(symlinks into the SDK cache; requires symlink support).')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print debug messages.')

    return parser


def _parse_args() -> argparse.Namespace:
    parser = _build_parser()
    import argcomplete
    argcomplete.autocomplete(parser)
    return parser.parse_args()


def _export_view(resources, output_dir: Path) -> None:
    """Create a browsable, project-scoped view of already-cached resource files."""
    for resource in resources:
        src = resource.filepath_cached
        if src is None:
            continue  # this one failed to prefetch, skip it
        dst = output_dir / resource.id / resource.filename
        if dst.exists() or dst.is_symlink():
            continue  # already linked from a previous run
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src, dst)
        except OSError as e:
            raise DatamintException(
                f"Could not create symlinks in '{output_dir}' ({e}). This usually means "
                "symlinks aren't supported here (e.g. Windows). ") from e


def main() -> None:
    load_cmdline_logging_config()

    args = _parse_args()

    if args.verbose:
        logging.getLogger().handlers[0].setLevel(logging.DEBUG)
        logging.getLogger('datamint').setLevel(logging.DEBUG)
        _LOGGER.setLevel(logging.DEBUG)
        _USER_LOGGER.setLevel(logging.DEBUG)

    try:
        api_key = handle_api_key()
        if api_key is None:
            _USER_LOGGER.error('API key not provided. Aborting.')
            sys.exit(1)

        from datamint import configs
        os.environ[configs.ENV_VARS[configs.APIKEY_KEY]] = api_key

        from datamint import build_dataset
        dataset = build_dataset(project=args.project)
        dataset.prefetch(include_annotations=True)

        _USER_LOGGER.info(f"✅ Downloaded {len(dataset.resources)} resource(s) for project '{args.project}'.")

        dataset_class_name = type(dataset).__name__
        _USER_LOGGER.info(
            f"\nUse it in your own scripts:\n\n"
            f"    from datamint import {dataset_class_name}\n\n"
            f"    dataset = {dataset_class_name}(project='{args.project}')\n"
        )

        if args.output:
            output_dir = Path(args.output).expanduser().resolve()
            _export_view(dataset.resources, output_dir)
            _USER_LOGGER.info(f"Browsable view created at: {output_dir}")

    except DatamintException as e:
        _USER_LOGGER.error(f'❌ {e}')
        sys.exit(1)
    except KeyboardInterrupt:
        _USER_LOGGER.warning('\nCancelled by user.')
        sys.exit(1)


if __name__ == '__main__':
    main()
