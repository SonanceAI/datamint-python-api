SSL Certificate Troubleshooting
================================

If you encounter SSL certificate verification errors like::

    SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate

Quick Fix
---------

**1. Upgrade certifi:**

.. code-block:: bash

    pip install --upgrade certifi

**2. Set environment variables:**

.. code-block:: bash

    export SSL_CERT_FILE=$(python -m certifi)
    export REQUESTS_CA_BUNDLE=$(python -m certifi)

**3. Run your script:**

.. code-block:: bash

    python your_script.py

Alternative Solutions
----------------------

**Option 1: Use Custom CA Bundle**

.. code-block:: python

    from datamint import Api

    api = Api(verify_ssl="/path/to/your/ca-bundle.crt")

**Option 2: Disable SSL Verification (Development Only)**

.. warning::

   Only use in development with self-signed certificates.

.. code-block:: python

    from datamint import Api

    api = Api(verify_ssl=False)
