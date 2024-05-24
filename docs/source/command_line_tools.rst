.. _command_line_tools:

Command-line tools
==================

To see if the Datamint command-line tool was installed correctly, you can run the
following command:

.. code-block:: bash

    datamint-upload --help

You should see something like this:

.. code-block:: bash

    usage: datamint-upload [-h] [-r [RECURSIVE]] [--mungfilename MUNGFILENAME] [--name NAME] [--retain-pii] [--retain-attribute RETAIN_ATTRIBUTE] [-l LABEL] --path FILE [--exclude EXCLUDE]
    (...)

Uploading DICOMs to Datamint server
-----------------------------------

To upload DICOM files to the Datamint server, use the
``datamint-upload`` command. For example, to upload all the DICOM files in the
``/path/to/dicom_files`` directory, run:

.. code-block:: bash

    datamint-upload --path /path/to/dicom_files

To upload all dicoms in a directory and also in its subdirectories,
you can use the recursive option ``-r`` flag:

.. code-block:: bash

    datamint-upload --path /path/to/dicom_files -r

To upload dicoms, associating them with a label, and associating them to a batch named "my_upload", run:

.. code-block:: bash

    datamint-upload --path /path/to/dicom_files -l "my_label" --name "my_upload"

See all available options by running ``datamint-upload --help``:

    -h, --help            show this help message and exit

    -r [RECURSIVE], --recursive [RECURSIVE]
                            Recurse folders looking for dicoms. If a number is passed, recurse that number of levels.

    --mungfilename MUNGFILENAME
                            Change the filename in the upload parameters. If set to "all", the filename becomes the folder names joined together with "_". If one or more integers are passed (comma-separated), append that
                            depth of folder name to the filename.
    --name NAME           Name of the upload batch
    --channel CHANNEL     Channel name (arbritary) to upload the dicoms to. Useful for organizing the dicoms in the platform.
    --retain-pii          Do not anonymize dicom
    --retain-attribute RETAIN_ATTRIBUTE
                            Retain the value of a single attribute code specified as hexidecimal integers. Example: (0x0008, 0x0050) or just (0008, 0050)
    -l LABEL, --label LABEL
                            A label name to be applied to all files
    --path FILE           Path to the DICOM file(s) or a directory
    --exclude EXCLUDE     Exclude folders that match the specified pattern. Example: "\*_not_to_upload" will exclude folders ending with "_not_to_upload
