.. _setup_api_key:

Setup API key
=============

There are three options to specify the API key:

1. **Recommended:** Run ``datamint-config`` in the terminal and follow the instructions. Run ``datamint-config --help`` for more optional information.
2. Specify API key as an environment variable:

.. tabs:: 

    .. code-tab:: bash

        export DATAMINT_API_KEY="my_api_key"
        python my_script.py

    .. code-tab:: python

        os.environ["DATAMINT_API_KEY"] = "my_api_key"
    
3. Specify API key in the |APIHandlerClass| constructor:

.. code-block:: python

   from datamintapi import APIHandler

   api = APIHandler(api_key='my_api_key')