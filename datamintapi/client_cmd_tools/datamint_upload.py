import argparse
from datamintapi.api_handler import APIHandler
import os
from humanize import naturalsize
import logging
from pathlib import Path
import sys
from datamintapi.utils.dicom_utils import is_dicom
import fnmatch
from typing import Sequence, List, Generator, Dict, Tuple, Optional, Any
from collections import defaultdict
from datamintapi import __version__ as datamintapi_version
from datamintapi import configs
from datamintapi.client_cmd_tools.datamint_config import ask_api_key
from datamintapi.utils.logging_utils import load_cmdline_logging_config
import yaml

# Create two loggings: one for the user and one for the developer
_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')

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


def walk_to_depth(path: str,
                  depth: int,
                  exclude_pattern: str = None) -> Generator[Path, None, None]:
    path = Path(path)
    for child in path.iterdir():
        if child.is_dir():
            if depth != 0:
                if exclude_pattern is not None and fnmatch.fnmatch(child.name, exclude_pattern):
                    continue
                yield from walk_to_depth(child, depth-1, exclude_pattern)
        else:
            _LOGGER.debug(f"yielding {child} from {path}")
            yield child


def filter_files(files_path: Sequence[Path],
                 include_extensions,
                 exclude_extensions) -> List[Path]:
    def fix_extension(ext: str) -> str:
        if ext == "" or ext[0] == '.':
            return ext
        return '.' + ext

    def normalize_extensions(exts_list: Sequence[str]) -> List[str]:
        # explodes the extensions if they are separated by commas
        exts_list = [ext.split(',') for ext in exts_list]
        exts_list = [item for sublist in exts_list for item in sublist]

        # adds a dot to the extensions if it does not have one
        exts_list = [fix_extension(ext) for ext in exts_list]

        return [fix_extension(ext) for ext in exts_list]

    if include_extensions is not None:
        include_extensions = normalize_extensions(include_extensions)
        files_path = [f for f in files_path if f.suffix in include_extensions]

    if exclude_extensions is not None:
        exclude_extensions = normalize_extensions(exclude_extensions)
        files_path = [f for f in files_path if f.suffix not in exclude_extensions]

    return files_path


def handle_api_key() -> str:
    """
    Checks for API keys.
    If it does not exist, it asks the user to input it.
    Then, it asks the user if he wants to save the API key at a proper location in the machine
    """
    api_key = configs.get_value(configs.APIKEY_KEY)
    if api_key is None:
        _USER_LOGGER.info("API key not found. Please provide it:")
        api_key = ask_api_key(ask_to_save=True)

    return api_key


def _find_segmentation_files(segmentation_root_path: str,
                             images_files: List[str],
                             segmentation_metainfo: Dict = None
                             ) -> Optional[List[Dict]]:
    """
    Find the segmentation files that match the images files based on the same folder structure
    """

    if segmentation_root_path is None:
        return None

    segmentation_files = []
    acceptable_extensions = ['.nii.gz', '.nii', '.png']

    if segmentation_metainfo is not None:
        segnames = sorted(segmentation_metainfo['segmentation_names'],
                          key=lambda x: len(x))
        classnames = segmentation_metainfo.get('class_names', None)
        _LOGGER.debug(f"Number of segmentation names: {len(segnames)}")
        if classnames is not None:
            _LOGGER.debug(f"Number of class names: {len(classnames)}")

    segmentation_root_path = Path(segmentation_root_path).absolute()

    for imgpath in images_files:
        imgpath_parent = Path(imgpath).absolute().parent
        # Find the closest common parent between the image and the segmentation root
        common_parent = []
        _LOGGER.debug(f'{imgpath_parent} | {segmentation_root_path.parent}')
        for imgpath_part, segpath_part in zip(imgpath_parent.parts, segmentation_root_path.parent.parts):
            if imgpath_part != segpath_part:
                break
            common_parent.append(imgpath_part)
        if len(common_parent) == 0:
            common_parent = Path('/')
        else:
            common_parent = Path(*common_parent)

        _LOGGER.debug(f"_find_segmentation_files::common_parent: {common_parent}")
        path_structure = imgpath_parent.relative_to(common_parent).parts[1:]

        # path_structure = imgpath_parent.relative_to(root_path).parts[1:]
        path_structure = Path(*path_structure)

        real_seg_root_path = common_parent / Path(Path(segmentation_root_path).relative_to(common_parent).parts[0])
        seg_path = real_seg_root_path / path_structure
        # list all segmentation files (nii.gz, nii, png) in the same folder structure
        seg_files = [fname for ext in acceptable_extensions for fname in seg_path.glob(f'*{ext}')]

        if len(seg_files) > 0:
            seginfo = {
                'files': [str(f) for f in seg_files]
            }
            if segmentation_metainfo is not None:
                snames_associated = []
                for segfile in seg_files:
                    for segname in segnames:
                        if segname in str(segfile):
                            if classnames is not None:
                                new_segname = {cid: f'{segname}_{cname}' for cid, cname in classnames.items()}
                                new_segname.update({'default': segname})
                            else:
                                new_segname = segname
                            snames_associated.append(new_segname)
                            break
                    else:
                        _USER_LOGGER.warning(f"Segmentation file {segname} does not match any segmentation name.")
                        snames_associated.append(None)
                seginfo['names'] = snames_associated

            segmentation_files.append(seginfo)
        else:
            segmentation_files.append(None)

    return segmentation_files


def _parse_args() -> Tuple[Any, List, Optional[List[Dict]]]:
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
    parser.add_argument('--channel', '--name', type=str, required=False,
                        help='Channel name (arbritary) to upload the resources to. \
                            Useful for organizing the resources in the platform.')
    parser.add_argument('--retain-pii', action='store_true', help='Do not anonymize DICOMs')
    parser.add_argument('--retain-attribute', type=_tuple_int_type, action='append',
                        default=[],
                        help='Retain the value of a single attribute code specified as hexidecimal integers. \
                            Example: (0x0008, 0x0050) or just (0008, 0050)')
    parser.add_argument('-l', '--label', type=str, action='append', help='A label name to be applied to all files')
    parser.add_argument('--publish', action='store_true',
                        help='Publish the uploaded resources, giving them the status "published" instead of "inbox"')
    parser.add_argument('--mungfilename', type=_mungfilename_type,
                        help='Change the filename in the upload parameters. \
                            If set to "all", the filename becomes the folder names joined together with "_". \
                            If one or more integers are passed (comma-separated), append that depth of folder name to the filename.')
    parser.add_argument('--include-extensions', type=str, nargs='+',
                        help='File extensions to be considered for uploading. Default: all file extensions.' +
                        ' Example: --include-extensions dcm jpg png')
    parser.add_argument('--exclude-extensions', type=str, nargs='+',
                        help='File extensions to be excluded from uploading. Default: none.' +
                        ' Example: --exclude-extensions txt csv'
                        )
    parser.add_argument('--segmentation_path', type=_is_valid_path_argparse, metavar="FILE",
                        required=False,
                        help='Path to the segmentation file(s) or a directory')
    parser.add_argument('--segmentation_names', type=_is_valid_path_argparse, metavar="FILE",
                        required=False,
                        help='Path to a yaml file containing the segmentation names.' +
                        ' The file may contain two keys: "segmentation_names" and "class_names".')
    parser.add_argument('--yes', action='store_true',
                        help='Automatically answer yes to all prompts')
    parser.add_argument('--version', action='version', version=f'%(prog)s {datamintapi_version}')
    parser.add_argument('--verbose', action='store_true', help='Print debug messages', default=False)
    args = parser.parse_args()
    if args.verbose:
        # Get the console handler and set to debug
        logging.getLogger().handlers[0].setLevel(logging.DEBUG)
        _LOGGER.setLevel(logging.DEBUG)
        _USER_LOGGER.setLevel(logging.DEBUG)

    if args.retain_pii and len(args.retain_attribute) > 0:
        raise ValueError("Cannot use --retain-pii and --retain-attribute together.")

    # include-extensions and exclude-extensions are mutually exclusive
    if args.include_extensions is not None and args.exclude_extensions is not None:
        raise ValueError("--include-extensions and --exclude-extensions are mutually exclusive.")

    try:

        if os.path.isfile(args.path):
            file_path = [args.path]
            if args.recursive is not None:
                _USER_LOGGER.warning("Recursive flag ignored. Specified path is a file.")
        else:
            try:
                recursive_depth = 0 if args.recursive is None else args.recursive
                file_path = walk_to_depth(args.path, recursive_depth, args.exclude)
                file_path = filter_files(file_path, args.include_extensions, args.exclude_extensions)
                file_path = list(map(str, file_path))  # from Path to str
            except Exception as e:
                _LOGGER.error(f'Error in recursive search: {e}')
                raise e

        if len(file_path) == 0:
            raise ValueError(f"No valid file was found in {args.path}")

        if args.segmentation_names is not None:
            with open(args.segmentation_names, 'r') as f:
                segmentation_names = yaml.safe_load(f)
        else:
            segmentation_names = None

        _LOGGER.debug(f'finding segmentations at {args.segmentation_path}')
        segmentation_files = _find_segmentation_files(args.segmentation_path,
                                                      file_path,
                                                      segmentation_metainfo=segmentation_names)

        _LOGGER.info(f"args parsed: {args}")

        api_key = handle_api_key()
        if api_key is None:
            _USER_LOGGER.error("API key not provided. Aborting.")
            sys.exit(1)
        os.environ[configs.ENV_VARS[configs.APIKEY_KEY]] = api_key

        return args, file_path, segmentation_files

    except Exception as e:
        if args.verbose:
            _LOGGER.exception(e)
        raise e


def print_input_summary(files_path: List[str],
                        args,
                        segfiles: Optional[List[Dict]],
                        include_extensions=None):
    ### Create a summary of the upload ###
    total_files = len(files_path)
    total_size = sum(os.path.getsize(file) for file in files_path)

    # Count number of files per extension
    ext_dict = defaultdict(int)
    for file in files_path:
        ext_dict[os.path.splitext(file)[1]] += 1

    # sorts the extensions by count
    ext_counts = [(ext, count) for ext, count in ext_dict.items()]
    ext_counts.sort(key=lambda x: x[1], reverse=True)

    _USER_LOGGER.info(f"Number of files to be uploaded: {total_files}")
    _USER_LOGGER.info(f"\t{files_path[0]}")
    if total_files >= 2:
        if total_files >= 3:
            _USER_LOGGER.info("\t(...)")
        _USER_LOGGER.info(f"\t{files_path[-1]}")
    _USER_LOGGER.info(f"Total size of the upload: {naturalsize(total_size)}")
    _USER_LOGGER.info(f"Number of files per extension:")
    for ext, count in ext_counts:
        if ext == '':
            ext = 'no extension'
        _USER_LOGGER.info(f"\t{ext}: {count}")
    if len(ext_counts) > 1 and include_extensions is None:
        _USER_LOGGER.warning("Multiple file extensions found!" +
                             " Make sure you are uploading the correct files.")

    if segfiles is not None:
        num_segfiles = sum([1 if seg is not None else 0 for seg in segfiles])
        msg = f"Number of images with an associated segmentation: " +\
            f"{num_segfiles} ({num_segfiles / total_files:.0%})"
        if num_segfiles == 0:
            _USER_LOGGER.warning(msg)
        else:
            _USER_LOGGER.info(msg)
        # count number of segmentations files with names
        if args.segmentation_names is not None and num_segfiles > 0:
            segnames_count = sum([1 if 'names' in seg else 0 for seg in segfiles if seg is not None])
            msg = f"Number of segmentations with associated name: " + \
                f"{segnames_count} ({segnames_count / num_segfiles:.0%})"
            if segnames_count == 0:
                _USER_LOGGER.warning(msg)
            else:
                _USER_LOGGER.info(msg)


def print_results_summary(files_path: List[str],
                          results: List[str | Exception]):
    # Check for failed uploads
    failure_files = [f for f, r in zip(files_path, results) if isinstance(r, Exception)]
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


def main():
    load_cmdline_logging_config()

    try:
        args, files_path, segfiles = _parse_args()
    except Exception as e:
        _USER_LOGGER.error(f'Error validating arguments. {e}')
        return

    print_input_summary(files_path,
                        args=args,
                        segfiles=segfiles,
                        include_extensions=args.include_extensions)

    if not args.yes:
        confirmation = input("Do you want to proceed with the upload? (y/n): ")
        if confirmation.lower() != "y":
            _USER_LOGGER.info("Upload cancelled.")
            return
    #######################################

    has_a_dicom_file = any(is_dicom(f) for f in files_path)

    api_handler = APIHandler()
    results = api_handler.upload_resources(channel=args.channel,
                                           files_path=files_path,
                                           labels=args.label,
                                           on_error='skip',
                                           anonymize=args.retain_pii == False and has_a_dicom_file,
                                           anonymize_retain_codes=args.retain_attribute,
                                           mung_filename=args.mungfilename,
                                           publish=args.publish,
                                           segmentation_files=segfiles,
                                           )
    _USER_LOGGER.info('Upload finished!')
    _LOGGER.debug(f"Number of results: {len(results)}")

    print_results_summary(files_path, results)


if __name__ == '__main__':
    main()
