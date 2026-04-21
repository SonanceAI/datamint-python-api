Dataset Classes
===============

The ``datamint.dataset`` module provides specialised PyTorch-compatible dataset
classes for different medical imaging modalities. Import them directly:

.. code-block:: python

   from datamint.dataset import ImageDataset, VolumeDataset, VideoDataset

Split Modes
-----------

All dataset classes inherit
:py:meth:`~datamint.dataset.base.DatamintBaseDataset.split`, which supports
three split modes:

- Local random splitting with ratio kwargs such as ``train=0.7``.
- Project-scoped split assignments resolved through :py:meth:`api.projects.get_splits() <datamint.api.endpoints.projects_api.ProjectsApi.get_splits>`.
- Legacy ``split:*`` resource tags, which remain available for backwards compatibility but are deprecated.

When you call ``split()`` without an explicit mode, the client chooses the
mode automatically:

- If ratio kwargs are provided, a local random split is used.
- If no ratios are provided and the dataset was loaded from a project, project-scoped splits are used.
- Otherwise, legacy ``split:*`` resource tags are used.

.. code-block:: python

    from datamint.dataset import ImageDataset

    dataset = ImageDataset(project="my-project", include_unannotated=True)

    # Project-backed datasets prefer project-scoped assignments.
    project_parts = dataset.split()

    # Persist and replay the exact historical snapshot later.
    snapshot = project_parts["train"].split_as_of_timestamp
    replayed_parts = dataset.split(as_of_timestamp=snapshot)

    # Force an ad hoc local split instead.
    local_parts = dataset.split(train=0.8, val=0.2, seed=42)

To override the automatic selection, pass ``use_project_splits=True`` or
``use_server_splits=True`` explicitly. ``use_server_splits`` is deprecated and
exists only for compatibility with older tag-based workflows.

Project-scoped splits require the dataset to be loaded from a project and must
not be combined with ratio kwargs. Each resolved subset records
``split_name``, ``split_source``, and, when applicable,
``split_as_of_timestamp`` so downstream training and MLflow lineage can reuse
the same split snapshot.

Base Classes
------------

.. automodule:: datamint.dataset.base
   :members:
   :show-inheritance:

.. automodule:: datamint.dataset.multiframe_dataset
   :members:
   :show-inheritance:

Specialised Datasets
--------------------

.. automodule:: datamint.dataset.image_dataset
   :members:
   :show-inheritance:

.. automodule:: datamint.dataset.volume_dataset
   :members:
   :show-inheritance:

.. automodule:: datamint.dataset.video_dataset
   :members:
   :show-inheritance:

Sliced Datasets
---------------

.. automodule:: datamint.dataset.sliced_dataset
   :members:
   :show-inheritance:

.. automodule:: datamint.dataset.sliced_video_dataset
   :members:
   :show-inheritance:

Legacy Classes (Deprecated)
----------------------------

.. deprecated::
   The classes below are kept for backwards compatibility and may be removed in a
   future release. Use :class:`~datamint.dataset.image_dataset.ImageDataset` or
   :class:`~datamint.dataset.volume_dataset.VolumeDataset` instead.

.. automodule:: datamint.dataset.dataset
   :members:
   :show-inheritance:

.. automodule:: datamint.dataset.base_dataset
   :members:

