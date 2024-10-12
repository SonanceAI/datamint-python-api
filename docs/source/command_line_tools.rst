.. _command_line_tools:

Command-line tools
==================

To see if the Datamint command-line tools were installed correctly, run:

.. code-block:: bash

    datamint-config --help

You should see this in the first line:

.. code-block:: bash

    usage: usage: datamint-config [-h] [--api-key API_KEY] [--default-url DEFAULT_URL] [-i]
    (...)

There are two command-line tools available:

- ``datamint-config``: To configure the Datamint API key and URL.
- ``datamint-upload``: To upload DICOM files to the Datamint server.

.. _configuring_datamint_settings:

Configuring the Datamint settings
---------------------------------

The ``datamint-config`` command-line tool is useful for configuring the Datamint API key and URL,
consequently avoiding the need to manually pass them as arguments or environment variables to the other commands later.

To manage Datamint configurations, just run 

.. code-block:: bash

    datamint-config

It starts an interactive prompt, guiding you through the configuration process.

To set the API key and URL without the interactive prompt, use the command-line options:

.. code-block:: bash

    datamint-config --api-key YOUR_API_KEY --url "https://stagingapi.datamint.io"

Uploading DICOMs/resources to Datamint server
---------------------------------------------

To upload DICOM files to the Datamint server, use the
``datamint-upload`` command. For example, to upload all the DICOM files in the
``/path/to/dicom_files`` directory, run:

.. code-block:: bash

    datamint-upload --path /path/to/dicom_files/

By default, the DICOM files are anonymized before uploading. If you want to
retain the personal identifiable information (PII) in the DICOM files, use the
``--retain-pii`` flag:

.. code-block:: bash

    datamint-upload --path /path/to/dicom_files/ --retain-pii

To upload all DICOMs in a directory and also in its subdirectories,
you can use the recursive option ``-r`` flag:

.. code-block:: bash

    datamint-upload --path /path/to/dicom_files/ -r

In Datamint, you can use channels to organize your DICOMs/resources.
In that case, use the ``--channel`` flag:

.. code-block:: bash

    datamint-upload --path /path/to/video.mp4 --channel "CT scans"

To upload resources, associating them with a label, and associating them to a batch named "my_upload", run:

.. code-block:: bash

    datamint-upload --path /path/to/dicom_files -l "my_label" --name "my_upload"

You can bypass the inbox/review and directly publish your resources with the ``--publish`` flag:

.. code-block:: bash

    datamint-upload --path /path/to/resource_file --publish


Example using include and exclude extensions options:
+++++++++++++++++++++++++++++++++++++++++++++++++++++

To upload only DICOM files, run:

.. code-block:: bash

    datamint-upload --path /root_dir --include-extensions dcm

To upload all files except the .txt and .csv files, run:

.. code-block:: bash

    datamint-upload --path /root_dir --exclude-extensions txt,csv

Uploading segmentations along with the resources
++++++++++++++++++++++++++++++++++++++++++++++++

To upload segmentations along with the resources, you can use

.. code-block:: bash

    datamint-upload --path data/OAI_CARE/dicoms/ -r --segmentation_path data/OAI_CARE/segmentations/ --publish

, where both "data/OAI_CARE/dicoms/" and "data/OAI_CARE/segmentations/" must obey the same folder structure.
Both folders and files can have arbritary names,
but if you want to provide the segmentation label names, the segmentation file names must contain the segmentation name and you must provide a yaml file like this one:

.. code-block:: yaml

    segmentation_names: ["Bones", "BoneHead", "BML"]
    class_names: {
        1: "Femur",
        2: "Tibia",
        4: "FC",
        8: "TC",
        16: "PAT"
    }

, where the `segmentation_names` are the names being that a segmentation files (for instance, BoneHead is in 'Case14_9587749__TSE_BoneHead_2.nii.gz') 
and the `class_names` is mapping the pixel values to the class names.
`class_names` is optional, so you can provide only the `segmentation_names` if you don't want to map the pixel values.
You can provide the segmentation names file with the `--segmentation_names` flag:

.. code-block:: bash
    
    datamint-upload --path data/OAI_CARE/dicoms/ -r --segmentation_path data/OAI_CARE/segmentations/ --segmentation_names segmentation_names.yaml --publish

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
    --path FILE           Path to the resource file(s) or a directory

    -r [RECURSIVE], --recursive [RECURSIVE]
                          Recurse folders looking for dicoms. If a number is passed, recurse that number of levels.

    --exclude EXCLUDE     Exclude folders that match the specified pattern. Example: "\*_not_to_upload" will exclude folders ending with "_not_to_upload
    --name NAME           Name of the upload batch
    --channel CHANNEL     Channel name (arbritary) to upload the resources to. Useful for organizing the resources in the platform.
    --retain-pii          Do not anonymize DICOMs
    --retain-attribute RETAIN_ATTRIBUTE
                            Retain the value of a single attribute code specified as hexidecimal integers. Example: (0x0008, 0x0050) or just (0008, 0050)
    -l LABEL, --label LABEL
                            A label name to be applied to all files
                            --publish             Publish the uploaded resources, giving them the status "published" instead of "inbox"
    --mungfilename MUNGFILENAME
                            Change the filename in the upload parameters. If set to "all", the filename becomes the folder names joined together with "_". If one or more integers are passed (comma-separated), append that
                            depth of folder name to the filename.
    --include-extensions INCLUDE_EXTENSIONS
                            File extensions to be considered for uploading. Default: all file extensions. Example: ``--include-extensions dcm jpg png``
    --exclude-extensions EXCLUDE_EXTENSIONS
                          File extensions to be excluded from uploading. Default: none. Example: ``--exclude-extensions txt csv``
    --segmentation_path FILE
                          Path to the segmentation file(s) or a directory
    --segmentation_names FILE
                          Path to a yaml file containing the segmentation names. The file may contain two keys: "segmentation_names" and "class_names".
    --yes                 Automatically answer yes to all prompts
    --version             show program's version number and exit