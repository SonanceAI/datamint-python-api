Dataset Classes
===============

The ``datamint.dataset`` module provides specialised PyTorch-compatible dataset
classes for different medical imaging modalities. Import them directly:

.. code-block:: python

   from datamint.dataset import ImageDataset, VolumeDataset, VideoDataset

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

