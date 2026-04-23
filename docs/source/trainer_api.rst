.. _trainer_api:


Trainer API
===========

The trainer layer in ``datamint.lightning`` packages the usual Lightning workflow into a
small number of task-focused entry points. A trainer can:

- build the dataset and datamodule for a Datamint project,
- choose task-specific default transforms, loss functions, and metrics,
- create the Lightning trainer, MLflow logger, and checkpoint callbacks,
- train and test the model, and
- optionally register the resulting model in MLflow.


Available Trainers
------------------

``UNetPPTrainer``
   The fastest way to train a 2-D semantic segmentation model with a sensible UNet++
   configuration. For pure 3-D volume projects, it automatically slices the volumes into
   2-D samples before training.

``SemanticSegmentation2DTrainer``
   A more explicit 2-D segmentation trainer when you want to control the model,
   transforms, or loss. It auto-detects whether the project contains 2-D images or 3-D
   volumes and accepts ``slice_axis=`` to override the inferred plane for volume projects.

``SemanticSegmentation3DTrainer``
   Slice-based semantic segmentation for projects of 3-D volumes.

``ImageClassificationTrainer``
   Image classification using a ``timm`` backbone.


Quick Start
-----------

.. code-block:: python

   from datamint.lightning import UNetPPTrainer

   trainer = UNetPPTrainer(
       project="BUSI_Segmentation",
       image_size=256,
       batch_size=16,
       max_epochs=20,
       accelerator="auto",
   )

   results = trainer.fit()
   print(results["test_results"])

The built-in trainer configures the dataset, datamodule, model, MLflow logger,
checkpointing, and evaluation loop for you. After ``fit()``, the resolved objects are also
available as ``trainer.dataset``, ``trainer.datamodule``, and ``trainer.model``.


Inputs, Splits, and Outputs
---------------------------

Each trainer accepts exactly one of:

- ``project=...`` to let Datamint build the dataset automatically, or
- ``dataset=...`` to reuse a dataset you already configured yourself.

For ``SemanticSegmentation2DTrainer`` and ``UNetPPTrainer``, project-backed dataset
resolution is automatic: pure 2-D image projects use ``ImageDataset``, while pure 3-D
volume projects are converted to ``SlicedVolumeDataset``. The slice plane is inferred from
volume spacing and shape when possible, and falls back to ``'axial'``. To force a plane,
pass ``slice_axis='coronal'`` or another supported axis when constructing the trainer.

When you train from a project, the trainer expects train/val/test split assignments to
exist for that project. If you need strict split reproducibility across runs, pass the
historical split snapshot timestamp through ``split_as_of_timestamp``.

.. code-block:: python

   trainer = UNetPPTrainer(
       project="BUSI_Segmentation",
       split_as_of_timestamp="2026-04-21T12:34:56Z",
   )

``fit()`` returns a dictionary with these keys:

``model``
   The trained model instance.

``test_results``
   The metrics returned by Lightning ``test()``.

If you only want evaluation, use ``test()`` instead:

.. code-block:: python

   test_metrics = trainer.test(register_model=False)

With ``register_model=True``, the trainer logs and registers the current model.


Passing Lightning Trainer Options
---------------------------------

Any extra keyword arguments that are not consumed by Datamint are forwarded to
``lightning.Trainer``.

.. code-block:: python

   trainer = UNetPPTrainer(
       project="BUSI_Segmentation",
       max_epochs=12,
       accelerator="gpu",
       devices=1,
       precision="16-mixed",
       log_every_n_steps=10,
       trainer_kwargs={"enable_progress_bar": True},
   )


Using an External Model Inside a Datamint Trainer
-------------------------------------------------

There are two supported patterns, and they are not equivalent.

Preferred: Subclass a Datamint Lightning Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to swap the network architecture but keep Datamint's loss wiring, metrics,
MLflow model behaviour, and deployment-friendly prediction methods, subclass
``SegmentationModule`` or ``ClassificationModule`` and pass the class object to ``model=``.

.. code-block:: python

   import segmentation_models_pytorch as smp
   from datamint.lightning import SemanticSegmentation2DTrainer
   from datamint.lightning.trainers.lightning_modules import SegmentationModule


   class DeepLabV3PlusModule(SegmentationModule):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, class_names=["benign", "malignant"], **kwargs)
           self.model = smp.DeepLabV3Plus(
               encoder_name="resnet50",
               encoder_weights="imagenet",
               in_channels=3,
               classes=2,
           )

       def forward(self, x):
           return self.model(x)


   trainer = SemanticSegmentation2DTrainer(
       project="BUSI_Segmentation",
       image_size=256,
       model=DeepLabV3PlusModule,
   )

   results = trainer.fit()

When you pass the class object instead of an instance, the trainer instantiates it and
injects task defaults through ``loss_fn=`` and ``metrics_factories=``. This is the easiest
way to plug an external architecture into the Datamint trainer workflow while keeping the
resulting model Datamint-compatible.

Fully Custom LightningModule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a plain ``lightning.LightningModule``, pass the instance through
``model=`` and implement the full training logic yourself.

.. code-block:: python

   import lightning as L
   import segmentation_models_pytorch as smp
   import torch
   from datamint.lightning import SemanticSegmentation2DTrainer


   class ExternalSegmentationModule(L.LightningModule):
       def __init__(self):
           super().__init__()
           self.model = smp.Unet(
               encoder_name="resnet34",
               encoder_weights="imagenet",
               in_channels=3,
               classes=2,
           )
           self.loss_fn = torch.nn.BCEWithLogitsLoss()

       def forward(self, x):
           return self.model(x)

       def training_step(self, batch, batch_idx):
           images = batch["image"]
           masks = batch["segmentations"][:, 1:].float()
           loss = self.loss_fn(self(images), masks)
           self.log("train/loss", loss)
           return loss

       def validation_step(self, batch, batch_idx):
           images = batch["image"]
           masks = batch["segmentations"][:, 1:].float()
           loss = self.loss_fn(self(images), masks)
           self.log("val/loss", loss)
           return loss

       def test_step(self, batch, batch_idx):
           images = batch["image"]
           masks = batch["segmentations"][:, 1:].float()
           loss = self.loss_fn(self(images), masks)
           self.log("test/loss", loss)
           return loss

       def configure_optimizers(self):
           return torch.optim.Adam(self.parameters(), lr=1e-4)


   trainer = SemanticSegmentation2DTrainer(
       project="BUSI_Segmentation",
       model=ExternalSegmentationModule(),
       max_epochs=5,
   )

This gives you the Datamint dataset, split handling, MLflow logger, and checkpointing, but
the model itself remains a plain Lightning module. That means Datamint-native inference and
deployment behaviour is not added automatically. If you want the trained artifact to behave
like a Datamint model, prefer the ``SegmentationModule`` / ``ClassificationModule`` route,
or wrap the final model in a ``DatamintModel`` afterwards.

.. note::

   **Class vs. Instance**

   ``model=MyModule`` and ``model=MyModule()`` are different:

   - Pass the **class** when you want the trainer to inject ``loss_fn`` and
     ``metrics_factories``.
   - Pass an **instance** when the module is already fully configured and owns its entire
     training logic.

.. caution::

   Segmentation batches expose masks in ``batch["segmentations"]`` and include the
   background channel at index 0.


Related Examples
----------------

- `BUSI trainer notebook <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/use_cases/segmentation_2d_trainer_BUSI_tutorial.ipynb>`_
- `External model deployment tutorial <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/external_model_deployment_tutorial.ipynb>`_