Datamint vs Raw PyTorch
=======================

This page shows how Datamint removes boilerplate from common medical-imaging
training workflows by comparing side-by-side a raw PyTorch / Lightning setup
with the equivalent Datamint code.  All examples target **2-D semantic
segmentation** (e.g. BUSI, skin-dataset, or liver-segmentation projects).

Workflow comparison at a glance
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - **Responsibility**
     - **Raw PyTorch / Lightning**
     - **Datamint**
   * - | 📂 **Data loading**
     - | Write a custom ``torch.utils.data.Dataset`` subclass that
       | reads files, decodes DICOM / NIfTI / PNG, 
       | applies augmentations, and returns aligned tensors.
     - | One line: :py:class:`~datamint.dataset.image_dataset.ImageDataset`
       | handles DICOM series, NIfTI, PNG, JPEG, and annotation parsing
       | automatically.
   * - | 📊 **Train / val / test splits**
     - | Manually partition resource IDs, tag them, 
       | or write a split function;
       | must track split manually to 
       | ensure reproducibility with seeds.
     - | :py:meth:`~datamint.dataset.base.DatamintBaseDataset.split` resolves
       | project-scoped splits (or falls back to legacy ``split:*`` tags) and
       | returns a snapshot timestamp you can replay later.
   * - | 🔌 **DataModule wiring**
     - | Implement ``lightning.pytorch.core.LightningDataModule`` with
       | ``prepare_data``, ``setup``, ``train_dataloader``, ``val_dataloader``,
       | ``test_dataloader``.
     - | :py:class:`~datamint.lightning.datamodule.DatamintDataModule` wraps any
       | :py:class:`~datamint.dataset.base.DatamintBaseDataset` and provides all
       | dataloaders out of the box.
   * - | 🧱 **Model scaffolding**
     - | Subclass ``lightning.pytorch.LightningModule``; write
       | ``forward``, ``training_step``, ``validation_step``, ``test_step``,
       | ``configure_optimizers``.
     - | :py:class:`~datamint.lightning.trainers.lightning_modules.SegmentationModule`
       | provides ``forward``, ``predict``, and ``predict_batch`` with
       | Datamint-native inference.  Loss and metrics are injected automatically
       | when you pass the class (not an instance) to ``model=``.
   * - | ⚙️ **Trainer configuration**
     - | Instantiate ``lightning.pytorch.trainer.Trainer`` with
       | ``max_epochs``, ``accelerator``, ``devices``, ``precision``, logger,
       | and checkpoint callbacks.
     - | :py:class:`~datamint.lightning.trainers.UNetPPTrainer` or
       | :py:class:`~datamint.lightning.trainers.SemanticSegmentation2DTrainer`
       | builds the dataset, datamodule, model, MLflow logger, and checkpoint
       | callbacks for you.  Extra kwargs are forwarded to
       | ``lightning.pytorch.trainer.Trainer``.
   * - | 📈 **Experiment tracking**
     - | Configure ``lightning.pytorch.loggers.MLFlowLogger`` or
       | ``lightning.pytorch.loggers.TensorBoardLogger``; log
       | hyper-parameters, metrics, and artifacts manually.
     - | MLflow is auto-configured on first import of
       | ``datamint.mlflow``.  The trainer logs metrics, hyper-parameters,
       | and checkpoints automatically; ``register_model=True`` registers the
       | final artifact in MLflow.
   * - | 🚀 **Model deployment**
     - | Export the checkpoint, write inference code, 
       | wrap it in an MLflow ``mlflow.pyfunc.PythonModel``,
       | and deploy to your serving platform.
     - | Trained model already a :py:class:`~datamint.mlflow.flavors.model.DatamintModel`
       | that can be deployed via the :py:class:`~datamint.api.endpoints.deploy_model_api.DeployModelApi` API.

Example 1 -- Dataset and split setup
-------------------------------------

.. tab-set::

   .. tab-item:: Raw PyTorch

      .. code-block:: python
         :linenos:
         :emphasize-lines: 8,11,14,20,27,34

         import os
         from pathlib import Path
         from typing import List, Tuple

         import albumentations as A
         import numpy as np
         import torch
         from PIL import Image
         from torch.utils.data import Dataset, DataLoader


         class SegmentationDataset(Dataset):
             """Manually wired dataset for 2-D segmentation."""

             def __init__(
                 self,
                 image_paths: List[Path],
                 mask_paths: List[Path],
                 transform: A.Compose,
             ):
                 self.image_paths = image_paths
                 self.mask_paths = mask_paths
                 self.transform = transform

             def __len__(self) -> int:
                 return len(self.image_paths)

             def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                 # -- File-type detection and decoding ---------------------------------
                 ext = self.image_paths[idx].suffix.lower()

                 if ext in (".dcm", ".dicom"):
                     # DICOM: requires pydicom to read, then handle pixel spacing,
                     # rescale slope/intercept, windowing, and multi-frame series.
                     import pydicom
                     ds = pydicom.dcmread(str(self.image_paths[idx]))
                     img = ds.pixel_array.astype(float)
                     # Interpret which shape dimension is the channel, which varies across dicoms and may require manual handling.
                     # (...) This is a common source of bugs

                 elif ext == ".nii" or ext == ".nii.gz":
                     # NIfTI: requires nibabel or nitransforms; handles 3-D volumes,
                     # custom affine transforms, and voxel-to-RAS mapping.
                     import nibabel as nib
                     nifti_img = nib.load(str(self.image_paths[idx]))
                     img = nifti_img.get_fdata().astype(np.float32)
                     # For 3-D volumes you must slice or process the full volume.
                     # This is non-trivial for segmentation tasks since you must ensure alignment with the mask and
                     # know which axis is the correct one to slice on.
                     # (...)

                 elif ext in (".png", ".jpg", ".jpeg"):
                     img = Image.open(self.image_paths[idx]).convert("RGB")
                     img = np.array(img).astype(np.float32)
                 else:
                     raise ValueError(f"Unsupported image format: {ext}")

                 # -- Mask loading (similar complexity for medical formats) -----------
                 mask_ext = self.mask_paths[idx].suffix.lower()
                 if mask_ext in (".nii", ".nii.gz"):
                     nifti_mask = nib.load(str(self.mask_paths[idx]))
                     mask = nifti_mask.get_fdata().astype(np.int64)
                 elif mask_ext in (".png",):
                     mask = np.array(Image.open(self.mask_paths[idx]).convert("L")).astype(np.int64)
                 else:
                     raise ValueError(f"Unsupported mask format: {mask_ext}")

                 # -- Augmentation -----------------------------------------------------
                 augmented = self.transform(image=img, mask=mask)
                 image = torch.from_numpy(augmented["image"]).float().permute(2, 0, 1)
                 mask = torch.from_numpy(augmented["mask"]).long()

                 return image, mask


         # -- Split logic (manual) -------------------------------------------
         all_paths = list(Path("data/images").iterdir())
         np.random.seed(42)
         np.random.shuffle(all_paths)
         train_paths = all_paths[:140]
         val_paths = all_paths[140:160]
         test_paths = all_paths[160:]

         # -- Training transform (with augmentations) ------------------------
         train_transform = A.Compose([
             A.HorizontalFlip(p=0.5),
             A.RandomBrightnessContrast(p=0.3),
             A.Normalize(),
         ])

         train_ds = SegmentationDataset(
             train_paths,
             [Path("data/masks") / p.name for p in train_paths],
             train_transform,
         )
         train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

         # -- Eval transform (no augmentations) ------------------------------
         eval_transform = A.Compose([A.Normalize()])

         val_ds = SegmentationDataset(
             val_paths,
             [Path("data/masks") / p.name for p in val_paths],
             eval_transform,
         )
         val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

         test_ds = SegmentationDataset(
             test_paths,
             [Path("data/masks") / p.name for p in test_paths],
             eval_transform,
         )
         test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

   .. tab-item:: Datamint

      .. code-block:: python
         :linenos:
         :emphasize-lines: 4,7,12

         import albumentations as A
         from datamint.dataset import ImageDataset
         from datamint.lightning import DatamintDataModule

         # 1. Load the project -- Datamint resolves all images and masks.
         dataset = ImageDataset(project="BUSI_Segmentation")

         # 2. Define transforms per stage.
         train_tfm = A.Compose([
             A.HorizontalFlip(p=0.5),
             A.RandomBrightnessContrast(p=0.3),
             A.Normalize(),
         ])
         eval_tfm = A.Compose([A.Normalize()])

         # 3. Wrap in a DataModule -- splitting and dataloaders are handled.
         dm = DatamintDataModule(
             dataset,
             batch_size=16,
             split={'train': 0.7, 'val': 0.1, 'test': 0.1},
             split_seed=42,
             train_transform=train_tfm,
             eval_transform=eval_tfm,
         )

         # Call prepare_data() / setup() before accessing dataloaders.
         dm.prepare_data()

         train_loader = dm.train_dataloader()
         val_loader = dm.val_dataloader()
         test_loader = dm.test_dataloader()

.. rst-class:: comparison-table

+-----------------------------------+------------------------------------------+------------------------------------------+
| **Aspect**                        | **Raw PyTorch**                          | **Datamint**                             |
+-----------------------------------+------------------------------------------+------------------------------------------+
| Dataset class                     | ``SegmentationDataset``                  | ``ImageDataset(project="...")``          |
|                                   | (30-50 lines PNG/JPEG)                   | *(1 line)*                               |
|                                   | (100-150 lines DICOM/NIfTI)              |                                          |
+-----------------------------------+------------------------------------------+------------------------------------------+
| Train/val/test splits             | Manual split logic + 3 instances         | ``split={'train': 0.7, ...}``            |
|                                   | (20-30 lines)                            | *(inline)*                               |
+-----------------------------------+------------------------------------------+------------------------------------------+
| DataModule                        | Custom ``LightningDataModule``           | ``DatamintDataModule(dataset, ...)``     |
|                                   | (40-60 lines)                            | *(1 line)*                               |
+-----------------------------------+------------------------------------------+------------------------------------------+
| **Total boilerplate**             | **~120-180 lines** (PNG/JPEG)            | **~5-10 lines**                          |
|                                   | **~200-300 lines** (DICOM/NIfTI)         |                                          |
+-----------------------------------+------------------------------------------+------------------------------------------+

Example 2 -- Full training loop
--------------------------------

.. tab-set::

   .. tab-item:: Raw PyTorch + Lightning

      .. code-block:: python
         :linenos:
         :emphasize-lines: 6,12,22,28,34,40

         import lightning as L
         import segmentation_models_pytorch as smp
         import torch


         class SegmentationModel(L.LightningModule):
             def __init__(self, num_classes: int = 2):
                 super().__init__()
                 self.model = smp.Unet(
                     encoder_name="resnet34",
                     encoder_weights="imagenet",
                     in_channels=3,
                     classes=num_classes,
                 )
                 self.loss_fn = torch.nn.CrossEntropyLoss()

             def forward(self, x):
                 return self.model(x)

             def training_step(self, batch, batch_idx):
                 images, masks = batch
                 logits = self(images)
                 loss = self.loss_fn(logits, masks)
                 self.log("train/loss", loss)
                 return loss

             def validation_step(self, batch, batch_idx):
                 images, masks = batch
                 logits = self(images)
                 loss = self.loss_fn(logits, masks)
                 self.log("val/loss", loss)
                 return loss

             def test_step(self, batch, batch_idx):
                 images, masks = batch
                 logits = self(images)
                 loss = self.loss_fn(logits, masks)
                 self.log("test/loss", loss)
                 return loss

             def configure_optimizers(self):
                 return torch.optim.Adam(self.parameters(), lr=1e-4)


         # -- Trainer setup --------------------------------------------------
         model = SegmentationModel(num_classes=2)
         trainer = L.Trainer(
             max_epochs=20,
             accelerator="auto",
             devices=1,
             precision="16-mixed",
             callbacks=[
                 L.pytorch.callbacks.ModelCheckpoint(
                     monitor="val/loss",
                     save_top_k=1,
                     mode="min",
                 )
             ],
             logger=L.pytorch.loggers.MLFlowLogger(experiment_name="BUSI_Segmentation"),
         )

         trainer.fit(model, datamodule=dm)
         trainer.test(model, datamodule=dm)

   .. tab-item:: Datamint

      .. code-block:: python
         :linenos:
         :emphasize-lines: 1,3,11

         from datamint.lightning import UNetPPTrainer

         trainer = UNetPPTrainer(
             project="BUSI_Segmentation",
             image_size=256,
             batch_size=16,
             max_epochs=20,
             accelerator="auto",
             # All extra kwargs are forwarded to lightning.Trainer.
             precision="16-mixed",
             devices=1,
         )

         results = trainer.fit()
         trainer.test()  # evaluates on the test split

         # The model is already registered in MLflow.
         print(results["test_results"])

.. rst-class:: comparison-table

+-----------------------------------+------------------------------------------+------------------------------------------+
| **Aspect**                        | **Raw PyTorch**                          | **Datamint**                             |
+-----------------------------------+------------------------------------------+------------------------------------------+
| Model definition                  | ``SegmentationModel`` (50-80 lines)      | Built internally by trainer              |
+-----------------------------------+------------------------------------------+------------------------------------------+
| Training step                     | Manual ``training_step`` with loss       | Automatic loss injection                 |
|                                   | logging                                  | via ``SegmentationModule``               |
+-----------------------------------+------------------------------------------+------------------------------------------+
| Validation step                   | Manual ``validation_step`` with loss     | Automatic                                |
+-----------------------------------+------------------------------------------+------------------------------------------+
| Test step                         | Manual ``test_step`` with loss           | Automatic                                |
+-----------------------------------+------------------------------------------+------------------------------------------+
| Optimizer                         | Manual ``configure_optimizers``          | Automatic                                |
+-----------------------------------+------------------------------------------+------------------------------------------+
| Trainer setup                     | ``Trainer`` + logger + callbacks         | Passed as kwargs to trainer              |
|                                   | (15-25 lines)                            |                                          |
+-----------------------------------+------------------------------------------+------------------------------------------+

.. note::
   Datamint's trainer handles dataset creation, datamodule wiring, model
   instantiation, MLflow logger setup, and checkpoint callbacks
   **automatically** — all from a single :py:class:`~datamint.lightning.trainers.UNetPPTrainer`
   call.

Example 3 -- Inference & deployment
------------------------------------

After training, Datamint models are **already registered in MLflow** and can be
loaded and used for inference with **zero extra code**.  The built-in
:py:class:`~datamint.mlflow.flavors.prediction_modes.PredictionMode` system
supports image, slice, volume, frame, and interactive prediction modes.

.. tab-set::

   .. tab-item:: Raw PyTorch (manual inference)

      .. code-block:: python
         :linenos:
         :emphasize-lines: 5,8,14,20,26

         import os
         from pathlib import Path
         import numpy as np
         import torch
         from PIL import Image
         import pydicom
         import nibabel as nib
         import mlflow


         # 1. Load model from MLflow ----------------------------------------
         model_uri = "runs:/abc123/artifacts/model"
         model = mlflow.pyfunc.load_model(model_uri)

         # 2. Manual preprocessing per format --------------------------------
         def preprocess_image(image_path: Path) -> torch.Tensor:
             img = Image.open(image_path).convert("RGB")
             img = np.array(img).astype(np.float32) / 255.0
             img = (img - 0.5) / 0.5  # manual normalization
             return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


         def preprocess_dicom(dicom_path: Path) -> torch.Tensor:
             ds = pydicom.dcmread(str(dicom_path))
             img = ds.pixel_array.astype(np.float32)
             # Handle rescale slope/intercept
             if hasattr(ds, "RescaleSlope"):
                 img = img * ds.RescaleSlope + ds.RescaleIntercept
             # Handle windowing
             if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                 wc = ds.WindowCenter[0] if hasattr(ds.WindowCenter, "__iter__") else ds.WindowCenter
                 ww = ds.WindowWidth[0] if hasattr(ds.WindowWidth, "__iter__") else ds.WindowWidth
                 img = (img - (wc - ww / 2)) / (ww / 2)
             img = np.clip(img, -1, 1)
             return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


         def preprocess_nifti(nifti_path: Path) -> torch.Tensor:
             nifti_img = nib.load(str(nifti_path))
             img = nifti_img.get_fdata().astype(np.float32)
             # Must manually handle 3-D → 2-D slicing
             img = img[:, :, img.shape[2] // 2]  # center slice
             img = (img - img.mean()) / (img.std() + 1e-8)
             return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


         # 3. Manual post-processing -----------------------------------------
         def postprocess_logits(logits: torch.Tensor) -> np.ndarray:
             probs = torch.softmax(logits, dim=1).cpu().numpy()
             mask = np.argmax(probs, axis=1).squeeze()
             return (mask * 255).astype(np.uint8)


         # 4. Run inference ---------------------------------------------------
         # For a PNG image:
         tensor = preprocess_image(Path("test_image.png"))
         logits = model.predict(tensor)
         result_mask = postprocess_logits(logits)

         # For a DICOM file (different preprocessing!):
         tensor = preprocess_dicom(Path("test_dicom.dcm"))
         logits = model.predict(tensor)
         result_mask = postprocess_logits(logits)

         # For a NIfTI volume (yet another preprocessing!):
         tensor = preprocess_nifti(Path("test_volume.nii.gz"))
         logits = model.predict(tensor)
         result_mask = postprocess_logits(logits)

   .. tab-item:: Datamint

      .. code-block:: python
        :linenos:
        :emphasize-lines: 3,5,8,11

        import mlflow
        from datamint.entities import LocalResource
        from datamint.mlflow.flavors import load_model

        # 1. Load model from MLflow -------------------
        model_uri = 'models:/UNetPP_Segmentation_Tutorial/latest'
        model = mlflow.pyfunc.load_model(model_uri)

        # 2. Predict on each format through the same API -------------------
        # For a PNG image:
        result = model.predict([LocalResource("test-image.png")])

        # For a DICOM file:
        result = model.predict([LocalResource("test_dicom.dcm")])

        # For a NIfTI volume (and slice prediction):
        model = load_model(model_uri)  # Use Datamint's MLflow flavor to get the extended API
        result = model.predict_slice(
            model_input=[LocalResource("test_dicom.nii.gz")],
            slice_index=10,
            axis="axial"
        )

        # 3. Results are Datamint annotations -- ready for platform use ----
        for annotation in result[0]:
            print(annotation)

.. note::
   The Datamint path eliminates **~70%** of the boilerplate code required by
   the raw PyTorch approach, especially when working with medical imaging formats
   like DICOM and NIfTI.


Key Observations
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - **Aspect**
     - **Raw PyTorch**
     - **Datamint**
   * - **Data loading**
     - Custom Dataset, DataModule, split logic
     - ``ImageDataset`` + ``DatamintDataModule``
   * - **Model definition**
     - Manual UNetPP implementation
     - Built-in ``UNetPPTrainer`` with defaults
   * - **Training loop**
     - Manually implemented
     - Automatic via Lightning ``Trainer``
   * - **Experiment tracking**
     - Manual MLflow logging
     - Automatic callback-based logging
   * - **Checkpoint management**
     - Manual export & naming
     - Automatic versioning + MLflow registration
   * - **Inference wrapper**
     - Custom code per model
     - Built-in ``PredictionMode`` system
   * - **Deployment**
     - Manual API integration
     - One-call ``api.deploy()``

At **inference time**, the key differences are:

.. list-table:: Inference-time comparison
   :header-rows: 1
   :widths: 28 36 36

   * - **Aspect**
     - **Raw PyTorch**
     - **Datamint**
   * - Input
     - Caller handles file I/O, decoding, normalization, and channel ordering.
     - Pass resource descriptors such as ``{"path": "...""}``; the wrapper
       resolves and preprocesses automatically.
   * - Output
     - Returns raw logits, probabilities, or masks requiring post-processing.
     - Returns ``list[list[Annotation]]`` ready for platform workflows.
   * - Multi-mode inference
     - Each new mode (slice, volume, frame …) requires custom glue code.
     - Controlled via ``PredictionMode``: ``IMAGE``, ``SLICE``, ``VOLUME``,
       ``FRAME``, ``FRAME_RANGE``, ``ALL_FRAMES``, ``TEMPORAL_SEQUENCE``.

Further reading
---------------

- :ref:`pytorch_integration` -- Dataset and datamodule details.
- :ref:`trainer_api` -- Training your Model reference and external-model patterns.
- `BUSI trainer notebook <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/06_end_to_end/slice_based/02_busi_segmentation.ipynb>`_
  -- Runnable end-to-end segmentation tutorial.
- `External model deployment tutorial <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/05_deployment/02_deploy_external_model.ipynb>`_
  -- Deploy a custom model trained with Datamint.
