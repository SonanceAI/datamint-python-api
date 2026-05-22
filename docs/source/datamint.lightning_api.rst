Lightning API
=============

The ``datamint.lightning`` module provides PyTorch Lightning integration for
training machine learning models with Datamint datasets.

.. toctree::
   :maxdepth: 2
   :caption: Lightning Components

   datamint.lightning.datamodule
   datamint.lightning.trainers
   datamint.lightning.lightning_modules

DatamintDataModule
------------------

The :py:class:`~datamint.lightning.datamodule.DatamintDataModule` wraps any
:py:class:`~datamint.dataset.base.DatamintBaseDataset` and provides
``train_dataloader``, ``val_dataloader``, ``test_dataloader``, and
``predict_dataloader`` for use with a Lightning
:class:`~lightning.pytorch.trainer.trainer.Trainer`.

.. code-block:: python

    import albumentations as A
    from datamint.dataset import ImageDataset
    from datamint.lightning import DatamintDataModule

    train_tfm = A.Compose([A.RandomHorizontalFlip(), A.Normalize()])
    eval_tfm  = A.Compose([A.Normalize()])

    dataset = ImageDataset(project='my_project')
    dm = DatamintDataModule(
        dataset,
        batch_size=8,
        split={'train': 0.8, 'val': 0.1, 'test': 0.1},
        split_seed=42,
        train_transform=train_tfm,
        eval_transform=eval_tfm,
    )

    trainer = lightning.Trainer(...)
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)

Constructor Parameters
++++++++++++++++++++++

.. automodule:: datamint.lightning.datamodule
   :members: __init__
   :undoc-members:

Key Methods
+++++++++++

.. automodule:: datamint.lightning.datamodule
   :members: prepare_data, setup, train_dataloader, val_dataloader, test_dataloader, predict_dataloader
   :undoc-members:
   :no-index:

Trainers
--------

The trainer layer packages the usual Lightning workflow into a small number of
task-focused entry points. A trainer can:

- Build the dataset and datamodule for a Datamint project,
- Choose task-specific default transforms, loss functions, and metrics,
- Create the Lightning trainer, MLflow logger, and checkpoint callbacks,
- Train and test the model, and
- Optionally register the resulting model in MLflow.

Available Trainers
++++++++++++++++++

.. automodule:: datamint.lightning.trainers
   :members:
   :undoc-members:
   :show-inheritance:

BaseTrainer
+++++++++++

.. automodule:: datamint.lightning.trainers.base_trainer
   :members:
   :undoc-members:
   :show-inheritance:

Segmentation Trainers
+++++++++++++++++++++

.. automodule:: datamint.lightning.trainers.seg2d_trainer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.lightning.trainers.seg3d_trainer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.lightning.trainers.segmentation_trainer
   :members:
   :undoc-members:
   :show-inheritance:

Classification Trainers
+++++++++++++++++++++++

.. automodule:: datamint.lightning.trainers.classification_trainer
   :members:
   :undoc-members:
   :show-inheritance:

Specialized Trainers
++++++++++++++++++++

.. automodule:: datamint.lightning.trainers.specialized.unetpp
   :members:
   :undoc-members:
   :show-inheritance:

Lightning Modules
-----------------

Subclass these modules to plug custom architectures into the Datamint trainer
workflow while keeping Datamint-native inference and deployment behavior.

.. automodule:: datamint.lightning.trainers.lightning_modules.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.lightning.trainers.lightning_modules.segmentation_module
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.lightning.trainers.lightning_modules.classification_module
   :members:
   :undoc-members:
   :show-inheritance:
