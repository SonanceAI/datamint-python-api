import argparse
from datamintapi._api_handler import APIHandler
import os
import argparse
import pydicom
from humanize import naturalsize

ROOT_URL = 'https://stagingapi.datamint.io'


def _is_valid_path_argparse(x):
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def _parse_args() -> tuple:
    parser = argparse.ArgumentParser(description='DatamintAPI command line tool for uploading dicom files')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recurse folders looking for dicoms')
    # TODO: discuss how exactly recursive should work
    parser.add_argument('--name', type=str, default='remote upload', help='Name of the upload batch')
    parser.add_argument('--retain-pii', action='store_true', help='Do not anonymize dicom')
    parser.add_argument('--retain-attribute', type=str, action='append',
                        help='Retain the value of a single attribute code')
    parser.add_argument('-l', '--label', type=str, action='append', help='A label name to be applied to all files')
    parser.add_argument('--path', type=_is_valid_path_argparse, metavar="FILE",
                        required=True,
                        help='Path to the DICOM file(s) or a directory')

    args = parser.parse_args()

    if os.path.isdir(args.path):
        file_path = [os.path.join(args.path, f)
                     for f in os.listdir(args.path) if f.endswith('.dcm') or f.endswith('.dicom')]
    else:
        file_path = [file_path]

    if len(file_path) == 0:
        raise ValueError(f"No dicom files found in {args.path}")

    return args, file_path


def main():
    try:
        args, files_path = _parse_args()
    except Exception as e:
        print(f"Error: {e}")
        return

    total_files = len(files_path)
    total_size = sum(os.path.getsize(file) for file in files_path)

    print(f"Number of DICOMs to be uploaded: {total_files}")
    print(f"\t{files_path[0]}")
    if total_files > 2:
        print("\t(...)")
    print(f"\t{files_path[-1]}")
    print(f"Total size of the upload: {naturalsize(total_size)}")

    confirmation = input("Do you want to proceed with the upload? (y/n): ")
    if confirmation.lower() != "y":
        print("Upload cancelled.")
        return

    # TODO: create a new dicom that has attributes anonymized

    api_handler = APIHandler(ROOT_URL, api_key='abc123')
    batch_id, _ = api_handler.create_new_batch(args.name,
                                               file_path=files_path,
                                               label=args.label)
    print('Upload finished!')
    batch_info = api_handler.get_batch_info(batch_id)
    batch_images = batch_info['images']
    all_images_filenames = [img['filename'] for img in batch_images]

    failure_files = []
    for fsubmitted in files_path:
        if os.path.basename(fsubmitted) not in all_images_filenames:
            # Should we only check for the basename?
            failure_files.append(fsubmitted)

    # Refine: Use colors here?
    print(f"\nUpload summary:")
    print(f"\tTotal files: {len(files_path)}")
    print(f"\tSuccessful uploads: {len(files_path) - len(failure_files)}")
    print(f"\tFailed uploads: {len(failure_files)}")
    if len(failure_files) > 0:
        print(f"\tFailed files: {failure_files}")


if __name__ == '__main__':
    main()
