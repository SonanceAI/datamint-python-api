Setup API Key
~~~~~~~~~~~~~

Obtaining an API key
--------------------

Before using the Datamint API, you need to configure your API key.
If you have the necessary permissions, you can obtain one from the `Datamint platform <https://app.datamint.io/>`_:

1. In the left sidebar, select **Teams**.
2. Click **Edit** on your user profile.

.. image:: ../images/how_to_get_api_key.png
   :align: center
   :alt: Navigating to the Teams page and editing your profile

|

3. Click **Generate API key** to create a new API key.

.. image:: ../images/how_to_get_api_key_2.png
   :align: center
   :alt: Generating an API key

.. note::

   If you don't have the necessary permissions, ask your administrator.

Configuring your API key
------------------------

Once you have your API key, choose one of the following options:

**Option 1: Using the CLI tool (recommended)**

.. code-block:: bash

    datamint-config --api-key YOUR_API_KEY

**Option 2: Setting an environment variable**

.. code-block:: bash

    export DATAMINT_API_KEY="your_api_key"

**Option 3: Programmatically in Python**

.. code-block:: python

    from datamint import Api

    api = Api(api_key="your_api_key")

