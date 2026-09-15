End-to-End Examples
====================

Complete pipelines from data upload to deployment on public datasets.

Each notebook below trains on a public dataset (BCCD, FracAtlas, BUSI, Synapse). The
download-and-upload step for each one is a single call into ``datamint.examples``, a
helper module that downloads the raw dataset, uploads it to a Datamint project, and
creates the matching annotations:

.. code-block:: python

   from datamint.examples import bccd_dataset

   project = bccd_dataset.create(project_name="bccd_detection")

Other available modules: ``fracatlas_dataset``, ``busi_dataset``, ``synapse_dataset``.

.. toctree::
   :maxdepth: 1

   slice_based
   full_3d
   sam
