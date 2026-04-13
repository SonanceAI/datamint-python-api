.. _command_line_tools:

Command-line tools
==================

To see if the Datamint command-line tools were installed correctly, run:

.. code-block:: bash

    datamint-config --help

You should see this in the first line:

.. code-block:: bash

    usage: datamint-config [-h] [--api-key API_KEY] [--default-url DEFAULT_URL] [-i] [...]
    (...)

There are two command-line tools available:

- ``datamint-config``: To configure the Datamint API key and URL.
- ``datamint-upload``: To upload DICOM, NIfTI, video, image, and segmentation files to the Datamint server.

.. _configuring_datamint_settings:

Configuring the Datamint settings
---------------------------------

The ``datamint-config`` command-line tool is useful for configuring the Datamint API key and URL,
consequently avoiding the need to manually pass them as arguments or environment variables to the other commands later.

To manage Datamint configurations, just run 

.. code-block:: bash

    datamint-config

It starts an interactive prompt, guiding you through the configuration process.

To set the API key without the interactive prompt, use the command-line option ``--api-key``:

.. code-block:: bash

    datamint-config --api-key YOUR_API_KEY

The same CLI can also inspect and clean local Datamint data stored under ``~/.datamint``.
Active local data is grouped by cache namespace such as ``resources`` or ``annotations``
instead of individual cached resources, so cleanup happens at that higher level:

.. code-block:: bash

    datamint-config --list-local-data
    datamint-config --clean-local-data resources
    datamint-config --clean-all-local-data

Uploading DICOMs/resources to Datamint server
---------------------------------------------

To upload DICOM files to the Datamint server, use the
``datamint-upload`` command. For example, to upload all the DICOM files in the
``/path/to/dicom_files`` directory, run:

.. code-block:: bash

    datamint-upload /path/to/dicom_files/

By default, the DICOM files are anonymized before uploading. If you want to
retain the personal identifiable information (PII) in the DICOM files, use the
``--retain-pii`` flag:

.. code-block:: bash

    datamint-upload /path/to/dicom_files/ --retain-pii

To upload all DICOMs in a directory and also in its subdirectories,
you can use the recursive option ``-r`` flag:

.. code-block:: bash

    datamint-upload /path/to/dicom_files/ -r

In Datamint, you can use channels to organize your DICOMs/resources.
In that case, use the ``--channel`` flag:

.. code-block:: bash

    datamint-upload /path/to/video.mp4 --channel "CT scans"

To upload resources, associating them with a tag, run:

.. code-block:: bash

    datamint-upload /path/to/dicom_files --tag "my_tag"

You can specify multiple tags by repeating the ``--tag`` flag:

.. code-block:: bash

    datamint-upload /path/to/dicom_files --tag "tag1" --tag "tag2"

You can bypass the inbox/review and directly publish your resources with the ``--publish`` flag:

.. code-block:: bash

    datamint-upload /path/to/resource_file --publish


Example using include and exclude extensions options:
+++++++++++++++++++++++++++++++++++++++++++++++++++++

To upload only DICOM files, run:

.. code-block:: bash

    datamint-upload /root_dir --include-extensions dcm

To upload all files except the .txt and .csv files, run:

.. code-block:: bash

    datamint-upload /root_dir --exclude-extensions txt csv

Uploading segmentations along with the resources
++++++++++++++++++++++++++++++++++++++++++++++++

To upload segmentations along with the resources, you can use

.. code-block:: bash

    datamint-upload data/OAI_CARE/dicoms/ -r --segmentation_path data/OAI_CARE/segmentations/ --publish

Both ``data/OAI_CARE/dicoms/`` and ``data/OAI_CARE/segmentations/`` must obey the same folder structure.
Both folders and files can have arbitrary names.

If you want to provide segmentation label names, use ``--segmentation_names`` with
either a YAML file or an ITK-SNAP label export CSV/TXT file. For YAML input, the
file can look like this:

.. code-block:: yaml

    segmentation_names: ["Bones", "BoneHead", "BML"]
    class_names: {
        1: "Femur",
        2: "Tibia",
        4: "FC",
        8: "TC",
        16: "PAT"
    }

Here, ``segmentation_names`` are matched from the segmentation filenames
(for instance, ``BoneHead`` in ``Case14_9587749__TSE_BoneHead_2.nii.gz``)
and ``class_names`` maps pixel values to class names.
``class_names`` is optional, so you can provide only ``segmentation_names`` if you do not want to map pixel values.
You can provide the segmentation names file with the ``--segmentation_names`` flag:

.. code-block:: bash
    
    datamint-upload data/OAI_CARE/dicoms/ -r --segmentation_path data/OAI_CARE/segmentations/ --segmentation_names segmentation_names.yaml --publish

Associating uploaded segmentations with a deployed model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

If the uploaded segmentations were produced by a deployed Datamint model, use
``--ai-model`` to associate the created annotations with that model:

.. code-block:: bash

    datamint-upload data/OAI_CARE/dicoms/ -r --segmentation_path data/OAI_CARE/segmentations/ --segmentation_names segmentation_names.yaml --ai-model knee-segmentation-v2 --publish

The value passed to ``--ai-model`` must match the name of an existing deployed
model on the server. This option only affects uploaded segmentations; resource
uploads without ``--segmentation_path`` are unchanged.

**JSON Metadata Support for NIfTI files:**

When uploading NIfTI files (.nii or .nii.gz), the tool automatically detects and includes JSON metadata files with the same base name. 
For example, if you have ``image.nii.gz``, it will automatically include ``image.json`` if it exists.

.. code-block:: bash

    datamint-upload /path/to/nifti_files/ -r

This feature can be disabled with ``--no-auto-detect-json`` flag:

.. code-block:: bash

    datamint-upload /path/to/nifti_files/ -r --no-auto-detect-json

To check if the segmentations were uploaded correctly, you can see some information after running your command line:

.. code-block:: console

    (...)
    Number of images with an associated segmentation: 4 (100%)                                                                                                                                                                                                      
    Number of segmentations with associated name: 4 (100%)   
    Do you want to proceed with the upload? (y/n): 

All available options
+++++++++++++++++++++

See all available options by running ``datamint-upload --help``:

  -h, --help            show this help message and exit
  --path FILE           Path to the resource file(s) or a directory (alternative to positional argument)
  -r [RECURSIVE], --recursive [RECURSIVE]
                        Recurse folders looking for DICOMs. If a number is passed, recurse that number of levels.
  --exclude EXCLUDE     Exclude folders that match the specified pattern. Example: "*_not_to_upload" will exclude folders ending with "_not_to_upload
  --channel CHANNEL, --name CHANNEL
                        Channel name (arbritary) to upload the resources to. Useful for organizing the resources in the platform.
  --project PROJECT     Project name to add the uploaded resources to after successful upload.
  --retain-pii          Do not anonymize DICOMs
  --retain-attribute RETAIN_ATTRIBUTE
                        Retain the value of a single attribute code specified as hexidecimal integers. Example: (0x0008, 0x0050) or just (0008, 0050)
  -l LABEL, --label LABEL
                        Deprecated. Use --tag instead.
  --tag TAG             A tag name to be applied to all files
  --publish             Publish the uploaded resources, giving them the status "published" instead of "inbox"
  --mungfilename MUNGFILENAME
                        Change the filename in the upload parameters. If set to "all", the filename becomes the folder names joined together with "_". If one or more
                        integers are passed (comma-separated), append that depth of folder name to the filename.
  --include-extensions INCLUDE_EXTENSIONS [INCLUDE_EXTENSIONS ...]
                        File extensions to be considered for uploading. Default: all file extensions. Example: --include-extensions dcm jpg png
  --exclude-extensions EXCLUDE_EXTENSIONS [EXCLUDE_EXTENSIONS ...]
                        File extensions to be excluded from uploading, in addition to the default exclusions. Default exclusions include common non-medical file
                        extensions (.txt, .json, .xml, .docx, etc.). Use --no-default-exclusions to disable the defaults. Example: --exclude-extensions txt csv
  --segmentation_path FILE
                        Path to the segmentation file(s) or a directory
  --segmentation_names FILE
                        Path to a yaml or csv file containing the segmentation names. If yaml, the file may contain two keys: "segmentation_names" and "class_names".
                        If csv, the file should be in itk-snap label export format, i.e, it should contain the following columns (with no header): index, r, g, b, ...,
                        name
  --ai-model MODEL_NAME
                        Name of a deployed AI model to associate with uploaded segmentations.
  --yes                 Automatically answer yes to all prompts
  --transpose-segmentation
                        Transpose the segmentation dimensions to match the image dimensions
  --auto-detect-json    Automatically detect and include JSON metadata files with the same base name as NIFTI files
  --no-auto-detect-json
                        Disable automatic detection of JSON metadata files (default behavior)
  --no-assemble-dicoms  Do not assemble DICOM files into series (default: assemble them)
  --no-default-exclusions
                        Disable the default excluded extensions list. By default, common non-medical file extensions are excluded unless --include-extensions is used.
  --version             show program's version number and exit
  --verbose             Print debug messages