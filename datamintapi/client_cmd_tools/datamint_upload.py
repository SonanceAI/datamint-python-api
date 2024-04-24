import importlib.resources
import logging.config
import argparse
from datamintapi._api_handler import APIHandler
import os
import argparse
from humanize import naturalsize
import logging
import yaml
import importlib
from netrc import netrc
from pathlib import Path
import sys

# Create two loggings: one for the user and one for the developer
_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')

ROOT_URL = 'https://stagingapi.datamint.io'


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


def _handle_api_key() -> str:
    """
    Checks for API keys in the env variable `DATAMINT_API_KEY`.
    If it does not exist, it asks the user to input it.
    Then, it asks the user if he wants to save the API key at a proper location in the machine 
    TODO: move this function to a separate module
    """
    api_key = os.getenv('DATAMINT_API_KEY')
    if api_key is None:
        try:
            netrc_file = Path.home() / ".netrc"
            if netrc_file.exists():
                token = netrc(netrc_file).authenticators('api.datamint.io')
                if token is not None:
                    return token[2]
            _USER_LOGGER.info("API key not found in enviroment variable DATAMINT_API_KEY. Please provide it:")
            api_key = input('API key (leave empty to abort): ').strip()
            if api_key == '':
                return None

            ans = input("Save the API key so it automatically loads next time? (y/n): ")
            try:
                if ans.lower() == 'y':
                    with open(netrc_file, 'a') as f:
                        f.write(f"\nmachine api.datamint.io\n  login user\n  password {api_key}\n")
                    _USER_LOGGER.info(f"API key saved to {netrc_file}")
            except Exception as e:
                _USER_LOGGER.error(f"Error saving API key.")
                _LOGGER.exception(e)
        except OSError as e:
            _USER_LOGGER.error(f"Error accessing netrc file. Contact your system administrator if you want to\
                               save the API key locally.")
            _LOGGER.exception(e)

    return api_key


def _parse_args() -> tuple:
    parser = argparse.ArgumentParser(description='DatamintAPI command line tool for uploading dicom files')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recurse folders looking for dicoms')
    # TODO: discuss how exactly recursive should work
    parser.add_argument('--name', type=str, default='remote upload', help='Name of the upload batch')
    parser.add_argument('--retain-pii', action='store_true', help='Do not anonymize dicom')
    parser.add_argument('--retain-attribute', type=_tuple_int_type, action='append',
                        help='Retain the value of a single attribute code specified as hexidecimal integers. \
                            Example: (0x0008, 0x0050) or just (0008, 0050)')
    parser.add_argument('-l', '--label', type=str, action='append', help='A label name to be applied to all files')
    parser.add_argument('--path', type=_is_valid_path_argparse, metavar="FILE",
                        required=True,
                        help='Path to the DICOM file(s) or a directory')

    args = parser.parse_args()

    if args.retain_pii is not None and args.retain_attribute is not None:
        raise ValueError("Cannot use --retain-pii and --retain-attribute together.")

    if args.retain_attribute is None:
        args.retain_attribute = []

    if os.path.isfile(args.path):
        file_path = [file_path]
    elif args.recursive == True:
        file_path = []
        for root, _, files in os.walk(args.path):
            for f in files:
                if f.endswith('.dcm') or f.endswith('.dicom'):
                    file_path.append(os.path.join(root, f))
    else:
        file_path = [os.path.join(args.path, f)
                     for f in os.listdir(args.path) if f.endswith('.dcm') or f.endswith('.dicom')]

    if len(file_path) == 0:
        raise ValueError(f"No dicom files found in {args.path}")

    _LOGGER.info(f"args parsed: {args}")

    api_key = _handle_api_key()
    if api_key is None:
        _USER_LOGGER.warning("API key not provided. Aborting.")
        sys.exit(1)
    os.environ['DATAMINT_API_KEY'] = api_key

    return args, file_path


def main():
    # Load the logging configuration file
    try:
        with importlib.resources.open_text('datamintapi', 'logging.yaml') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except Exception as e:
        print("Warning: Error loading logging configuration file.")
        logging.basicConfig(level=logging.INFO)

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

    api_handler = APIHandler(ROOT_URL)
    batch_id, _ = api_handler.create_new_batch(args.name,
                                               file_path=files_path,
                                               labels=args.label,
                                               anonymize=args.retain_pii == False,
                                               anonymize_retain_codes=args.retain_attribute
                                               )
    _USER_LOGGER.info('Upload finished!')
    batch_info = api_handler.get_batch_info(batch_id)
    batch_images = batch_info['images']
    all_images_filenames = [img['filename'] for img in batch_images]

    failure_files = []
    for fsubmitted in files_path:
        if os.path.basename(fsubmitted) not in all_images_filenames:
            # Should we only check for the basename?
            failure_files.append(fsubmitted)

    # Refine: Use colors here?
    _USER_LOGGER.info(f"\nUpload summary:")
    _USER_LOGGER.info(f"\tTotal files: {len(files_path)}")
    _USER_LOGGER.info(f"\tSuccessful uploads: {len(files_path) - len(failure_files)}")
    _USER_LOGGER.info(f"\tFailed uploads: {len(failure_files)}")
    if len(failure_files) > 0:
        _USER_LOGGER.warning(f"\tFailed files: {failure_files}")


if __name__ == '__main__':
    main()
