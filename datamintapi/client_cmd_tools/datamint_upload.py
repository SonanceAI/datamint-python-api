import argparse
from datamintapi.api_handler import APIHandler
import os
import argparse
from humanize import naturalsize
import logging
from pathlib import Path
import sys
from datamintapi.utils.dicom_utils import is_dicom
import fnmatch
from datamintapi import configs
from datamintapi.client_cmd_tools.datamint_config import ask_api_key
from datamintapi.utils.logging_utils import load_cmdline_logging_config

# Create two loggings: one for the user and one for the developer
_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')

ROOT_URL = 'https://stagingapi.datamint.io'

MAX_RECURSION_LIMIT = 1000


def _is_valid_path_argparse(x):
    """
    argparse type that checks if the path exists
    """
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def _tuple_int_type(x: str):
    """
    argparse type that converts a string of two hexadecimal integers to a tuple of integers
    """
    try:
        x_processed = tuple(int(i, 16) for i in x.strip('()').split(','))
        if len(x_processed) != 2:
            raise ValueError
        return x_processed
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Values must be two hexadecimal integers separated by a comma. Example (0x0008, 0x0050)"
        )


def _mungfilename_type(arg):
    if arg.lower() == 'all':
        return 'all'
    try:
        ret = list(map(int, arg.split(',')))
        # can only have positive values
        if any(i <= 0 for i in ret):
            raise ValueError
        return ret
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid value for --mungfilename. Expected 'all' or comma-separated positive integers.")


def _walk_to_depth(path: str, depth: int, exclude_pattern: str = None):
    path = Path(path)
    for child in path.iterdir():
        if child.is_dir() and depth != 0:
            if exclude_pattern is not None and fnmatch.fnmatch(child.name, exclude_pattern):
                continue
            yield from _walk_to_depth(child, depth-1, exclude_pattern)
        else:
            yield child


def handle_api_key() -> str:
    """
    Checks for API keys in the env variable `DATAMINT_API_KEY`.
    If it does not exist, it asks the user to input it.
    Then, it asks the user if he wants to save the API key at a proper location in the machine
    """
    api_key = os.getenv('DATAMINT_API_KEY')
    if api_key is None:
        try:
            token = configs.get_value(configs.APIKEY_KEY)
            if token is not None:
                _LOGGER.info("API key loaded from netrc file.")
                return token[2]
            _USER_LOGGER.info("API key not found. Please provide it:")
            api_key = ask_api_key(True)
            if api_key is None:
                return None
        except OSError as e:
            _USER_LOGGER.error(f"Error accessing netrc file. Contact your system administrator if you want to\
                               save the API key locally.")
            _LOGGER.exception(e)

    return api_key


def _parse_args() -> tuple:
    parser = argparse.ArgumentParser(
        description='DatamintAPI command line tool for uploading DICOM files and other resources')
    parser.add_argument('--path', type=_is_valid_path_argparse, metavar="FILE",
                        required=True,
                        help='Path to the resource file(s) or a directory')
    parser.add_argument('-r', '--recursive', nargs='?', const=-1,  # -1 means infinite
                        type=int,
                        help='Recurse folders looking for DICOMs. If a number is passed, recurse that number of levels.')
    parser.add_argument('--exclude', type=str,
                        help='Exclude folders that match the specified pattern. \
                            Example: "*_not_to_upload" will exclude folders ending with "_not_to_upload')
    parser.add_argument('--name', type=str, help='Name of the upload batch.')
    parser.add_argument('--channel', type=str, required=False,
                        help='Channel name (arbritary) to upload the resources to. \
                            Useful for organizing the resources in the platform.')
    parser.add_argument('--retain-pii', action='store_true', help='Do not anonymize DICOMs')
    parser.add_argument('--retain-attribute', type=_tuple_int_type, action='append',
                        default=[],
                        help='Retain the value of a single attribute code specified as hexidecimal integers. \
                            Example: (0x0008, 0x0050) or just (0008, 0050)')
    parser.add_argument('-l', '--label', type=str, action='append', help='A label name to be applied to all files')
    parser.add_argument('--mungfilename', type=_mungfilename_type,
                        help='Change the filename in the upload parameters. \
                            If set to "all", the filename becomes the folder names joined together with "_". \
                            If one or more integers are passed (comma-separated), append that depth of folder name to the filename.')

    # parser.add_argument('--formats', type=str, nargs='+', default=['dcm', 'dicom'],
    #                     help='File extensions to be considered for uploading. Default: dcm, dicom')

    args = parser.parse_args()

    if args.retain_pii and len(args.retain_attribute) > 0:
        raise ValueError("Cannot use --retain-pii and --retain-attribute together.")

    if os.path.isfile(args.path):
        file_path = [args.path]
        if args.recursive is not None:
            _USER_LOGGER.warning("Recursive flag ignored. Specified path is a file.")
    elif args.recursive is not None:
        file_path = []
        for file in _walk_to_depth(args.path, args.recursive, args.exclude):
            if is_dicom(file):
                file_path.append(str(file))
    else:
        file_path = [os.path.join(args.path, f)
                     for f in os.listdir(args.path) if is_dicom(os.path.join(args.path, f))]

    if len(file_path) == 0:
        raise ValueError(f"No DICOM files found in {args.path}")

    _LOGGER.info(f"args parsed: {args}")

    api_key = handle_api_key()
    if api_key is None:
        _USER_LOGGER.error("API key not provided. Aborting.")
        sys.exit(1)
    os.environ['DATAMINT_API_KEY'] = api_key

    return args, file_path


def _verify_files_batch(files_path, api_handler, batch_id) -> list:
    """
    Verify if the files in the batch_id are the same as the files in the results
    """
    _LOGGER.debug(f'batch_id: {batch_id}')
    batch_info = api_handler.get_batch_info(batch_id)
    batch_images = batch_info['images']
    all_images_paths = [img['filepath'] for img in batch_images]

    failure_files = []
    for fsubmitted in files_path:
        if fsubmitted not in all_images_paths:
            failure_files.append(fsubmitted)

    return failure_files


def main():
    load_cmdline_logging_config()

    try:
        args, files_path = _parse_args()
    except Exception as e:
        _USER_LOGGER.error(e)
        return

    ### Create a summary of the upload ###
    total_files = len(files_path)
    total_size = sum(os.path.getsize(file) for file in files_path)

    _USER_LOGGER.info(f"Number of DICOMs to be uploaded: {total_files}")
    _USER_LOGGER.info(f"\t{files_path[0]}")
    if total_files >= 2:
        if total_files >= 3:
            _USER_LOGGER.info("\t(...)")
        _USER_LOGGER.info(f"\t{files_path[-1]}")
    _USER_LOGGER.info(f"Total size of the upload: {naturalsize(total_size)}")

    confirmation = input("Do you want to proceed with the upload? (y/n): ")
    if confirmation.lower() != "y":
        _USER_LOGGER.info("Upload cancelled.")
        return
    #######################################

    has_a_dicom_file = any(is_dicom(f) for f in files_path)

    api_handler = APIHandler(ROOT_URL)
    if args.name is not None:
        batch_id, results = api_handler.create_batch_with_dicoms(args.name,
                                                                 files_path=files_path,
                                                                 labels=args.label,
                                                                 on_error='skip',
                                                                 anonymize=args.retain_pii == False and has_a_dicom_file,
                                                                 anonymize_retain_codes=args.retain_attribute,
                                                                 mung_filename=args.mungfilename,
                                                                 channel=args.channel
                                                                 )
        _LOGGER.debug(f"new Batch ID: {batch_id}")
    else:
        results = api_handler.upload_resources(channel=args.channel,
                                               files_path=files_path,
                                               labels=args.label,
                                               on_error='skip',
                                               anonymize=args.retain_pii == False and has_a_dicom_file,
                                               anonymize_retain_codes=args.retain_attribute,
                                               mung_filename=args.mungfilename
                                               )
    _USER_LOGGER.info('Upload finished!')
    _LOGGER.debug(f"Number of results: {len(results)}")

    ### Check for failed uploads ###
    failure_files = [f for f, r in zip(files_path, results) if isinstance(r, Exception)]
    #################################

    # Refine: Use colors here?
    _USER_LOGGER.info(f"\nUpload summary:")
    _USER_LOGGER.info(f"\tTotal files: {len(files_path)}")
    _USER_LOGGER.info(f"\tSuccessful uploads: {len(files_path) - len(failure_files)}")
    _USER_LOGGER.info(f"\tFailed uploads: {len(failure_files)}")
    if len(failure_files) > 0:
        _USER_LOGGER.warning(f"\tFailed files: {failure_files}")
        _USER_LOGGER.warning(f"\nFailures:")
        for f, r in zip(files_path, results):
            _LOGGER.debug(f"Failure: {f} - {r}")
            if isinstance(r, Exception):
                _USER_LOGGER.warning(f"\t{os.path.basename(f)}: {r}")


if __name__ == '__main__':
    main()
