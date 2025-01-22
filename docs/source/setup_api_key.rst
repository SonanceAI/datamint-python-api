.. _setup_api_key:

Setup API key
=============

There are three options to specify the API key:

1. **Recommended:** Run ``datamint-config`` and follow the instructions. See :ref:`configuring_datamint_settings` for more details.
2. Specify API key as an environment variable:

.. tabs:: 

    .. code-tab:: bash

        export DATAMINT_API_KEY="my_api_key"
        python my_script.py

    .. code-tab:: python

        os.environ["DATAMINT_API_KEY"] = "my_api_key"
    
1. Specify API key in the |APIHandlerClass| constructor:

.. code-block:: python

   from datamintapi import APIHandler

   api_handler = APIHandler(api_key='my_api_key')