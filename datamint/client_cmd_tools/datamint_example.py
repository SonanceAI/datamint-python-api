"""datamint example command-line tool.

Populates a Datamint project with a small, ready-to-use example dataset -
no need to bring your own data first. See ``datamint example --help`` for
the list of available datasets.
"""
import argparse
import logging
import sys

from datamint.client_cmd_tools.datamint_upload import handle_api_key
from datamint.exceptions import DatamintException
from datamint.utils.logging_utils import load_cmdline_logging_config

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')

_DATASETS = ('bccd', 'busi', 'synapse', 'fracatlas')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Populate a Datamint project with an example dataset.',
        epilog="""
Examples:
  datamint example bccd                         # Blood cell detection (BCCD)
  datamint example busi --project MyBusiProject  # Breast ultrasound segmentation (BUSI)
  datamint example synapse                       # Multi-organ CT segmentation (Synapse)
  datamint example fracatlas                     # Fracture classification (FracAtlas)

More Documentation: https://sonanceai.github.io/datamint-python-api/command_line_tools.html
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('dataset', choices=_DATASETS, help='Which example dataset to populate.')
    parser.add_argument('--project', type=str, default=None,
                        help='Name of the project to create. Defaults to a dataset-specific name.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print debug messages.')

    return parser.parse_args()


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
        import os

        from datamint import configs
        os.environ[configs.ENV_VARS[configs.APIKEY_KEY]] = api_key

        from datamint import examples
        module = getattr(examples, f'{args.dataset}_dataset')
        kwargs = {'project_name': args.project} if args.project else {}
        module.create(**kwargs)
    except DatamintException as e:
        _USER_LOGGER.error(f'❌ {e}')
        sys.exit(1)
    except KeyboardInterrupt:
        _USER_LOGGER.warning('\nCancelled by user.')
        sys.exit(1)


if __name__ == '__main__':
    main()
