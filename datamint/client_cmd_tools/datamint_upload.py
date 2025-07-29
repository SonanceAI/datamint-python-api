import argparse
from datamint.apihandler.api_handler import APIHandler
import os
from humanize import naturalsize
import logging
from pathlib import Path
import sys
from medimgkit.dicom_utils import is_dicom
import fnmatch
from typing import Generator, Optional, Any
from collections import defaultdict
from datamint import __version__ as datamint_version
from datamint import configs
from datamint.client_cmd_tools.datamint_config import ask_api_key
from datamint.utils.logging_utils import load_cmdline_logging_config
import yaml
from collections.abc import Iterable
import pandas as pd

# Create two loggings: one for the user and one for the developer
_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')

MAX_RECURSION_LIMIT = 1000


def _read_segmentation_names(segmentation_names_path: str | Path) -> dict:
    """
    Read a segmentation names file (yaml or csv) and return its content as a dictionary.
    If the file is a YAML file, it should contain two keys: "segmentation_names" and "class_names".
    If the file is a CSV file, it should contain the following columns:
    index, r, g, b, ..., name
    """
    segmentation_names_path = Path(segmentation_names_path)
    if segmentation_names_path.suffix in ['.yaml', '.yml']:
        with open(segmentation_names_path, 'r') as f:
            metadata = yaml.safe_load(f)
    elif segmentation_names_path.suffix in ['.csv', '.tsv']:
        df = pd.read_csv(segmentation_names_path,
                         header=None,
                         index_col=0,
                         sep=None,  # use sep=None to automatically detect the separator
                         engine='python'
                         )
        df = df.rename(columns={1: 'r', 2: 'g', 3: 'b', df.columns[-1]: 'name'})
        # df = df.set_index(['r', 'g', 'b'])
        metadata = {'class_names': df['name'].to_dict()}
    else:
        raise ValueError(f"Unsupported file format: {segmentation_names_path.suffix}")

    if 'segmentation_names' in metadata:
        segnames = sorted(metadata['segmentation_names'],
                          key=lambda x: len(x))
        metadata['segmentation_names'] = segnames

    return metadata


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


def _is_system_file(path: Path) -> bool:
    """
    Check if a file is a system file that should be ignored
    """
    # Common system files and folders to ignore
    ignored_patterns = [
        '.DS_Store',
        'Thumbs.db',
        '.git',
        '__pycache__',
        '*.pyc',
        '.svn',
        '.tmp',
        '~*',  # Temporary files created by some editors
        '._*'  # macOS resource fork files
    ]

    # Check if path is inside a system folder
    system_folders = ['__MACOSX', '$RECYCLE.BIN', 'System Volume Information']
    if any(folder in path.parts for folder in system_folders):
        return True

    # Check if filename matches any ignored pattern
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in ignored_patterns)


def walk_to_depth(path: str | Path,
                  depth: int,
                  exclude_pattern: str | None = None) -> Generator[Path, None, None]:
    path = Path(path)
    for child in path.iterdir():
        if _is_system_file(child):
            continue

        if child.is_dir():
            if depth != 0:
                if exclude_pattern is not None and fnmatch.fnmatch(child.name, exclude_pattern):
                    continue
                yield from walk_to_depth(child, depth-1, exclude_pattern)
        else:
            yield child


def filter_files(files_path: Iterable[Path],
                 include_extensions: Optional[list[str]] = None,
                 exclude_extensions: Optional[list[str]] = None) -> list[Path]:
    def fix_extension(ext: str) -> str:
        if ext == "" or ext[0] == '.':
            return ext
        return '.' + ext

    def normalize_extensions(exts_list: Iterable[str]) -> list[str]:
        # explodes the extensions if they are separated by commas
        exts_list = [ext.split(',') for ext in exts_list]
        exts_list = [item for sublist in exts_list for item in sublist]

        # adds a dot to the extensions if it does not have one
        exts_list = [fix_extension(ext) for ext in exts_list]

        return [fix_extension(ext) for ext in exts_list]

    files_path = list(files_path)
    # Filter out files less than 4 bytes
    files_path2 = [f for f in files_path if f.stat().st_size >= 4]
    if len(files_path) != len(files_path2):
        _USER_LOGGER.info(f"Filtered out {len(files_path) - len(files_path2)} empty files")
    files_path = files_path2

    if include_extensions is not None:
        include_extensions = normalize_extensions(include_extensions)
        files_path = [f for f in files_path if f.suffix in include_extensions]

    if exclude_extensions is not None:
        exclude_extensions = normalize_extensions(exclude_extensions)
        files_path = [f for f in files_path if f.suffix not in exclude_extensions]

    return files_path


def handle_api_key() -> str | None:
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
                             images_files: list[str],
                             segmentation_metainfo: dict | None = None
                             ) -> list[dict]:
    """
    Find the segmentation files that match the images files based on the same folder structure
    """

    segnames = None
    classnames = None
    if segmentation_metainfo is not None:
        if 'segmentation_names' in segmentation_metainfo:
            segnames = sorted(segmentation_metainfo['segmentation_names'],
                              key=lambda x: len(x))
        classnames = segmentation_metainfo.get('class_names', None)
        if classnames is not None:
            _LOGGER.debug(f"Number of class names: {len(classnames)}")

    if len(images_files) == 1 and os.path.isfile(images_files[0]) and os.path.isfile(segmentation_root_path):
        ret = [{'files': [segmentation_root_path]}]
        if classnames is not None:
            ret[0]['names'] = classnames
        _LOGGER.debug(f"Returning segmentation files: {ret}")
        return ret

    segmentation_files = []
    acceptable_extensions = ['.nii.gz', '.nii', '.png']

    segmentation_root_path = Path(segmentation_root_path).absolute()

    for imgpath in images_files:
        imgpath_parent = Path(imgpath).absolute().parent
        # Find the closest common parent between the image and the segmentation root
        common_parent = []
        for imgpath_part, segpath_part in zip(imgpath_parent.parts, segmentation_root_path.parent.parts):
            if imgpath_part != segpath_part:
                break
            common_parent.append(imgpath_part)
        if len(common_parent) == 0:
            common_parent = Path('/')
        else:
            common_parent = Path(*common_parent)

        path_structure = imgpath_parent.relative_to(common_parent).parts[1:]

        # path_structure = imgpath_parent.relative_to(root_path).parts[1:]
        path_structure = Path(*path_structure)

        real_seg_root_path = common_parent / Path(Path(segmentation_root_path).relative_to(common_parent).parts[0])
        seg_path = real_seg_root_path / path_structure
        # list all segmentation files (nii.gz, nii, png) in the same folder structure
        seg_files = [fname for ext in acceptable_extensions for fname in seg_path.glob(f'*{ext}')]
        if len(seg_files) == 0:
            filename = Path(imgpath).stem
            seg_path = seg_path / filename
            seg_files = [fname for ext in acceptable_extensions for fname in seg_path.glob(f'*{ext}')]

        if len(seg_files) > 0:
            seginfo = {
                'files': [str(f) for f in seg_files]
            }

            frame_indices = []
            for segfile in seg_files:
                if segfile.suffix == '.png':
                    try:
                        frame_index = int(segfile.stem)
                    except ValueError:
                        frame_index = None

                    frame_indices.append(frame_index)

            if len(frame_indices) > 0:
                seginfo['frame_index'] = frame_indices

            snames_associated = []
            for segfile in seg_files:
                # check if there is a metadata file associated, besides json, with the segmentation
                for ext in ['.yaml', '.yml', '.csv']:
                    if str(segfile).endswith('nii.gz'):
                        # has two extensions, so we need to remove both
                        metadata_file = segfile.with_suffix('').with_suffix(ext)
                        if not metadata_file.exists():
                            metadata_file = segfile.with_suffix(ext)
                    else:
                        metadata_file = segfile.with_suffix(ext)
                    if metadata_file.exists():
                        _LOGGER.debug(f"Found metadata file: {metadata_file}")
                        try:
                            new_segmentation_metainfo = _read_segmentation_names(metadata_file)
                            cur_segnames = new_segmentation_metainfo.get('segmentation_names', segnames)
                            cur_classnames = new_segmentation_metainfo.get('class_names', classnames)
                            break
                        except Exception as e:
                            _LOGGER.warning(f"Error reading metadata file {metadata_file}: {e}")
                else:
                    cur_segnames = segnames
                    cur_classnames = classnames

                if cur_segnames is None:
                    _LOGGER.debug(f'adding {cur_classnames}')
                    snames_associated.append(cur_classnames)
                else:
                    for segname in cur_segnames:
                        if segname in str(segfile):
                            if cur_classnames is not None:
                                new_segname = {cid: f'{segname}_{cname}' for cid, cname in cur_classnames.items()}
                                new_segname.update({'default': segname})
                            else:
                                new_segname = segname
                            snames_associated.append(new_segname)
                            break
                    else:
                        _USER_LOGGER.warning(f"Segmentation file {segfile} does not match any segmentation name.")
                        snames_associated.append(None)
            if len(snames_associated) > 0:
                seginfo['names'] = snames_associated

            segmentation_files.append(seginfo)
        else:
            segmentation_files.append(None)

    return segmentation_files


def _find_json_metadata(file_path: str | Path) -> Optional[str]:
    """
    Find a JSON file with the same base name as the given file.

    Args:
        file_path (str): Path to the main file (e.g., NIFTI file)

    Returns:
        Optional[str]: Path to the JSON metadata file if found, None otherwise
    """
    file_path = Path(file_path)

    # Handle .nii.gz files specially - need to remove both extensions
    if file_path.name.endswith('.nii.gz'):
        base_name = file_path.name[:-7]  # Remove .nii.gz
        json_path = file_path.parent / f"{base_name}.json"
    else:
        json_path = file_path.with_suffix('.json')

    if json_path.exists() and json_path.is_file():
        _LOGGER.debug(f"Found JSON metadata file: {json_path}")
        return str(json_path)

    return None


def _collect_metadata_files(files_path: list[str], auto_detect_json: bool) -> tuple[list, list[str]]:
    """
    Collect JSON metadata files for the given files and filter them from main files list.

    Args:
        files_path (list[str]): List of file paths
        auto_detect_json (bool): Whether to auto-detect JSON metadata files

    Returns:
        tuple[list[Optional[str]], list[str]]: Tuple of (metadata file paths, filtered files_path)
            - metadata file paths: List of metadata file paths (None if no metadata found)
            - filtered files_path: Original files_path with JSON metadata files removed
    """
    if not auto_detect_json:
        return [None] * len(files_path), files_path

    metadata_files = []
    used_json_files = set()
    nifti_extensions = ['.nii', '.nii.gz']

    for file_path in files_path:
        # Check if this is a NIFTI file
        if any(file_path.endswith(ext) for ext in nifti_extensions):
            json_file = _find_json_metadata(file_path)
            metadata_files.append(json_file)
            if json_file is not None:
                used_json_files.add(json_file)
        else:
            metadata_files.append(None)

    # Filter out JSON files that are being used as metadata from the main files list
    filtered_files_path = [f for f in files_path if f not in used_json_files]

    # Update metadata_files to match the filtered list
    if used_json_files:
        _LOGGER.debug(f"Filtering out {len(used_json_files)} JSON metadata files from main upload list")
        filtered_metadata_files = []

        for original_file in files_path:
            if original_file not in used_json_files:
                original_index = files_path.index(original_file)
                filtered_metadata_files.append(metadata_files[original_index])

        metadata_files = filtered_metadata_files

    return metadata_files, filtered_files_path


def _parse_args() -> tuple[Any, list[str], Optional[list[dict]], Optional[list[str]]]:
    parser = argparse.ArgumentParser(
        description='DatamintAPI command line tool for uploading DICOM files and other resources')

    # Add positional argument for path
    parser.add_argument('path', nargs='?', type=_is_valid_path_argparse, metavar="PATH",
                        help='Path to the resource file(s) or a directory')

    # Keep the --path option for backward compatibility, but make it optional
    parser.add_argument('--path', dest='path_flag', type=_is_valid_path_argparse, metavar="FILE",
                        help='Path to the resource file(s) or a directory (alternative to positional argument)')
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
    parser.add_argument('-l', '--label', type=str, action='append', help='Deprecated. Use --tag instead.')
    parser.add_argument('--tag', type=str, action='append', help='A tag name to be applied to all files')
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
                        help='Path to a yaml or csv file containing the segmentation names.' +
                        ' If yaml, the file may contain two keys: "segmentation_names" and "class_names".'
                        ' If csv, the file should contain the following columns:'
                        ' index, r, g, b, ..., name')
    parser.add_argument('--yes', action='store_true',
                        help='Automatically answer yes to all prompts')
    parser.add_argument('--transpose-segmentation', action='store_true', default=False,
                        help='Transpose the segmentation dimensions to match the image dimensions')
    parser.add_argument('--auto-detect-json', action='store_true', default=True,
                        help='Automatically detect and include JSON metadata files with the same base name as NIFTI files')
    parser.add_argument('--no-auto-detect-json', dest='auto_detect_json', action='store_false',
                        help='Disable automatic detection of JSON metadata files (default behavior)')
    parser.add_argument('--version', action='version', version=f'%(prog)s {datamint_version}')
    parser.add_argument('--verbose', action='store_true', help='Print debug messages', default=False)
    args = parser.parse_args()

    # Handle path argument priority: positional takes precedence over --path flag
    if args.path is not None and args.path_flag is not None:
        _USER_LOGGER.warning("Both positional path and --path flag provided.")
        raise ValueError("Both positional path and --path flag provided.")
    elif args.path is not None and isinstance(args.path, (str, Path)):
        final_path = args.path
    elif args.path_flag is not None and isinstance(args.path_flag, (str, Path)):
        final_path = args.path_flag
    else:
        parser.error("Path argument is required. Provide it as a positional argument or use --path flag.")

    # Replace args.path with the final resolved path for consistency
    args.path = final_path

    if args.verbose:
        # Get the console handler and set to debug
        print(logging.getLogger().handlers)
        logging.getLogger().handlers[0].setLevel(logging.DEBUG)
        logging.getLogger('datamint').setLevel(logging.DEBUG)
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
                if recursive_depth < 0:
                    recursive_depth = MAX_RECURSION_LIMIT
                else:
                    recursive_depth = min(MAX_RECURSION_LIMIT, recursive_depth)
                file_path = walk_to_depth(args.path, recursive_depth, args.exclude)
                file_path = filter_files(file_path, args.include_extensions, args.exclude_extensions)
                file_path = list(map(str, file_path))  # from Path to str
            except Exception as e:
                _LOGGER.error(f'Error in recursive search: {e}')
                raise e

        if len(file_path) == 0:
            raise ValueError(f"No valid file was found in {args.path}")

        # Collect JSON metadata files and filter them from main files list
        metadata_files, file_path = _collect_metadata_files(file_path, args.auto_detect_json)

        if len(file_path) == 0:
            raise ValueError(f"No valid non-metadata files found in {args.path}")

        if args.segmentation_names is not None:
            segmentation_names = _read_segmentation_names(args.segmentation_names)
        else:
            segmentation_names = None

        _LOGGER.debug(f'finding segmentations at {args.segmentation_path}')
        if args.segmentation_path is None:
            segmentation_files = None
        else:
            segmentation_files = _find_segmentation_files(args.segmentation_path,
                                                          file_path,
                                                          segmentation_metainfo=segmentation_names)

        _LOGGER.info(f"args parsed: {args}")

        api_key = handle_api_key()
        if api_key is None:
            _USER_LOGGER.error("API key not provided. Aborting.")
            sys.exit(1)
        os.environ[configs.ENV_VARS[configs.APIKEY_KEY]] = api_key

        if args.tag is not None and args.label is not None:
            raise ValueError("Cannot use both --tag and --label. Use --tag instead. --label is deprecated.")
        args.tag = args.tag if args.tag is not None else args.label

        return args, file_path, segmentation_files, metadata_files

    except Exception as e:
        if args.verbose:
            _LOGGER.exception(e)
        raise e


def print_input_summary(files_path: list[str],
                        args,
                        segfiles: Optional[list[dict]],
                        metadata_files: Optional[list[str]] = None,
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
    # Check for multiple extensions
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

    if metadata_files is not None:
        num_metadata_files = sum([1 if metadata is not None else 0 for metadata in metadata_files])
        if num_metadata_files > 0:
            msg = f"Number of files with JSON metadata: {num_metadata_files} ({num_metadata_files / total_files:.0%})"
            _USER_LOGGER.info(msg)
            # TODO: Could add validation to ensure JSON metadata files contain valid DICOM metadata structure


def print_results_summary(files_path: list[str],
                          results: list[str | Exception]) -> int:
    # Check for failed uploads
    failure_files = [f for f, r in zip(files_path, results) if isinstance(r, Exception)]
    _USER_LOGGER.info(f"\nUpload summary:")
    _USER_LOGGER.info(f"\tTotal files: {len(files_path)}")
    _USER_LOGGER.info(f"\tSuccessful uploads: {len(files_path) - len(failure_files)}")
    _USER_LOGGER.info(f"\tFailed uploads: {len(failure_files)}")
    if len(failure_files) > 0:
        _USER_LOGGER.warning(f"\tFailed files: {[os.path.basename(f) for f in failure_files]}")
        _USER_LOGGER.warning(f"\nFailures:")
        for f, r in zip(files_path, results):
            _LOGGER.debug(f"Failure: {f} - {r}")
            if isinstance(r, Exception):
                _USER_LOGGER.warning(f"\t{os.path.basename(f)}: {r}")
    return len(failure_files)


def main():
    load_cmdline_logging_config()

    try:
        args, files_path, segfiles, metadata_files = _parse_args()
    except Exception as e:
        _USER_LOGGER.error(f'Error validating arguments. {e}')
        sys.exit(1)

    print_input_summary(files_path,
                        args=args,
                        segfiles=segfiles,
                        metadata_files=metadata_files,
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
                                           tags=args.tag,
                                           on_error='skip',
                                           anonymize=args.retain_pii == False and has_a_dicom_file,
                                           anonymize_retain_codes=args.retain_attribute,
                                           mung_filename=args.mungfilename,
                                           publish=args.publish,
                                           segmentation_files=segfiles,
                                           transpose_segmentation=args.transpose_segmentation,
                                           assemble_dicoms=True,
                                           metadata=metadata_files
                                           )
    _USER_LOGGER.info('Upload finished!')
    _LOGGER.debug(f"Number of results: {len(results)}")

    num_failures = print_results_summary(files_path, results)
    if num_failures > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
