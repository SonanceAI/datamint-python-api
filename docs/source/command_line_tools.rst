.. _command_line_tools:

Command-line tools
==================

To see if the Datamint command-line tool was installed correctly, you can run the
following command:

.. code-block:: bash

    datamint-upload --help

You should see this in the first line:

.. code-block:: bash

    usage: datamint-upload [-h] --path FILE [-r [RECURSIVE]] [--exclude EXCLUDE] [--name NAME] [--channel CHANNEL] [--retain-pii] [--retain-attribute RETAIN_ATTRIBUTE] [-l LABEL] [--mungfilename MUNGFILENAME]
    (...)

Uploading DICOMs/resources to Datamint server
---------------------------------------------

To upload DICOM files to the Datamint server, use the
``datamint-upload`` command. For example, to upload all the DICOM files in the
``/path/to/dicom_files`` directory, run:

.. code-block:: bash

    datamint-upload --path /path/to/dicom_files/

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
    --mungfilename MUNGFILENAME
                            Change the filename in the upload parameters. If set to "all", the filename becomes the folder names joined together with "_". If one or more integers are passed (comma-separated), append that
                            depth of folder name to the filename.
