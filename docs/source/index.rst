.. DatamintAPI documentation master file


.. image:: ../images/logo.png
   :height: 250
   :align: center
   :alt: Datamint

Documentation
=======================================
**Version:** |release|

From inception to completion, Datamint is your reliable partner.
It assists from the very first day when you make your data available to your team, right up to the moment you're set to launch your model.

Datamint
--------

- `Homepage <https://www.datamint.io>`_
- `Datamint Platform <https://app.datamint.io/>`_
- `Github <https://github.com/SonanceAI/datamint-python-api>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   command_line_tools
   client_api
   pytorch_integration


.. toctree::
   :maxdepth: 1
   :caption: Python Modules Reference

   datamint.apihandler
   datamint.dataset
   datamint.entities


Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install datamint

Configure your API access:

.. code-block:: bash

   datamint-config

Start using the API:

.. code-block:: python

   from datamint import Api
   
   # Initialize API handler
   api = Api()
   all_projects = api.projects.get_all()
   

Community & Support
-------------------
`GitHub Issues <https://github.com/SonanceAI/datamint-python-api/issues>`_

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
