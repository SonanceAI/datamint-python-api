.. _pytorch_integration:


PyTorch & Lightning Integration
===============================

The Datamint Python API provides seamless integration with PyTorch and PyTorch Lightning, enabling efficient machine learning workflows for medical imaging tasks.

Overview
--------

Key integration features:

- **DatamintDataModule**: Lightning-compatible data module
- **MLFlowModelCheckpoint**: Advanced model checkpointing with MLflow integration
- **Automatic Experiment Tracking**: Seamless logging and model registration
- **Medical Image Optimizations**: Specialized handling for medical data formats

PyTorch Dataset Integration
---------------------------

Basic PyTorch Usage
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from datamintapi import Dataset
   
   # Load dataset. This is a PyTorch-compatible dataset that can be used directly.
   dataset = Dataset(
       project_name="liver-segmentation",
       return_annotations=True,
       return_frame_by_frame=True,
       include_unannotated=False
   )
   
   # Create PyTorch DataLoader
   dataloader = DataLoader(
       dataset,
       batch_size=16,
       shuffle=True,
       num_workers=4,
       collate_fn=dataset.get_collate_fn()
   )
   
   # Training loop
   for batch in dataloader:
       images = batch['image']      # Shape: [B, C, H, W]
       masks = batch['segmentation'] # Shape: [B, H, W]
       metadata = batch['metainfo']  # List of dicts
       # (...)

Dataset Transforms
~~~~~~~~~~~~~~~~~~

Apply transforms for data augmentation and preprocessing:

.. code-block:: python

   import albumentations as A
   from albumentations.pytorch import ToTensorV2
   
   # Define image transforms
   image_transform = A.Compose([
       A.Resize(512, 512),
       A.Normalize(mean=[0.485], std=[0.229]),
       ToTensorV2()
   ])
   
   # Define mask transforms
   mask_transform = A.Compose([
       A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
       ToTensorV2()
   ])
   
   # Combined transforms (recommended for paired image-mask data)
   alb_transform = A.Compose([
       A.Resize(512, 512),
       A.RandomRotate90(p=0.5),
       A.Flip(p=0.5),
       A.RandomBrightnessContrast(p=0.3),
       A.Normalize(mean=[0.485], std=[0.229]),
       ToTensorV2()
   ], additional_targets={'mask': 'mask'})
   
   # Apply transforms to dataset
   dataset = Dataset(
       project_name="liver-segmentation",
       image_transform=image_transform,
       mask_transform=mask_transform,
       # OR use combined transforms:
       # alb_transform=alb_transform
   )

PyTorch Lightning Integration
-----------------------------

DatamintDataModule
~~~~~~~~~~~~~~~~~~

The ``DatamintDataModule`` provides a Lightning-compatible interface for Datamint datasets:

.. code-block:: python

   import lightning as L
   from datamint.lightning import DatamintDataModule
   import albumentations as A
   from albumentations.pytorch import ToTensorV2
   
   # Define transforms
   transforms = A.Compose([
       A.Resize(256, 256),
       A.RandomRotate90(p=0.5),
       A.Flip(p=0.5),
       A.Normalize(mean=[0.485], std=[0.229]),
       ToTensorV2()
   ], additional_targets={'mask': 'mask'})
   
   # Create data module
   datamodule = DatamintDataModule(
       project_name="medical-segmentation",
       batch_size=16,
       train_split=0.8,
       val_split=0.2,
       alb_transform=transforms,
       num_workers=4,
       seed=42
   )
   
   # Use with Lightning Trainer
   trainer = L.Trainer()
   trainer.fit(model, datamodule)

Custom Lightning Module Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import lightning as L
   from torchmetrics import Dice
   
   class SegmentationModel(L.LightningModule):
       def __init__(self, num_classes=2, learning_rate=1e-3):
           super().__init__()
           self.save_hyperparameters()
           
           # Simple U-Net-like architecture
           self.encoder = nn.Sequential(
               nn.Conv2d(1, 64, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 128, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2)
           )
           
           self.decoder = nn.Sequential(
               nn.ConvTranspose2d(128, 64, 2, stride=2),
               nn.ReLU(),
               nn.Conv2d(64, num_classes, 1)
           )
           
           # Metrics
           self.train_dice = Dice(num_classes=num_classes)
           self.val_dice = Dice(num_classes=num_classes)
           
       def forward(self, x):
           # x shape: [B, C, H, W]
           encoded = self.encoder(x)  # [B, 128, H/2, W/2]
           decoded = self.decoder(encoded)  # [B, num_classes, H, W]
           return decoded
           
       def training_step(self, batch, batch_idx):
           images = batch['image']  # [B, C, H, W]
           masks = batch['segmentation']  # [B, H, W]
           
           # Forward pass
           logits = self(images)  # [B, num_classes, H, W]
           
           # Calculate loss
           loss = F.cross_entropy(logits, masks.long())
           
           # Calculate metrics
           preds = torch.argmax(logits, dim=1)
           dice = self.train_dice(preds, masks)
           
           # Logging
           self.log('train_loss', loss, on_step=True, on_epoch=True)
           self.log('train_dice', dice, on_step=False, on_epoch=True)
           
           return loss
           
       def validation_step(self, batch, batch_idx):
           images = batch['image']
           masks = batch['segmentation']
           
           logits = self(images)
           loss = F.cross_entropy(logits, masks.long())
           
           preds = torch.argmax(logits, dim=1)
           dice = self.val_dice(preds, masks)
           
           self.log('val_loss', loss, on_epoch=True)
           self.log('val_dice', dice, on_epoch=True)
           
       def configure_optimizers(self):
           return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

MLflow Integration
------------------

The Datamint API provides enhanced MLflow integration for experiment tracking and model management.

Basic MLflow Setup
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import datamint.mlflow as dm_mlflow
   from lightning.pytorch.loggers import MLFlowLogger
   
   # Set up Datamint project for MLflow
   dm_mlflow.set_project(project_name="liver-segmentation")
   
   # Create MLflow logger
   logger = MLFlowLogger(experiment_name="segmentation-experiments",)

MLFlowModelCheckpoint Callback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MLFlowModelCheckpoint`` callback extends Lightning's ``ModelCheckpoint`` with automatic MLflow model logging:

.. code-block:: python

   from datamint.mlflow.lightning.callbacks import MLFlowModelCheckpoint
   
   # Advanced model checkpoint with MLflow integration
   checkpoint_callback = MLFlowModelCheckpoint(
       # Standard ModelCheckpoint parameters
       monitor="val_dice",
       mode="max",
       save_top_k=1,
       filename="best-model-{epoch:02d}-{val_dice:.3f}",
       
       # MLflow-specific parameters
       register_model_name="liver-segmentation-model",
       register_model_on="train",  # Register after training
       log_model_at_end_only=True,  # Log model only at end (more efficient)
       code_paths=["models/", "utils/"],  # Include your custom source code
       extra_pip_requirements=[
           "albumentations>=1.3.0",
           "pydicom>=2.3.0"
       ],
       additional_metadata={ # (Optional) Additional metadata necessary for deployment to cloud inference.
            "task_type": "semantic_segmentation",
            "labels": ["background", "liver", "tumor"],
            "need_gpu": False,                    # Whether GPU is required for inference
            "automatic_preprocessing": True       # Whether preprocessing is handled automatically
       }
   )

Complete Training Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import lightning as L
   from lightning.pytorch.loggers import MLFlowLogger
   from datamint.lightning import DatamintDataModule
   from datamint.mlflow.lightning.callbacks import MLFlowModelCheckpoint
   import datamint.mlflow as dm_mlflow
   
   # Set up Datamint project for MLflow
   dm_mlflow.set_project(project_name="liver-segmentation")
   
   # Data module
   datamodule = DatamintDataModule(
       project_name="liver-segmentation",
       batch_size=16,
       train_split=0.8,
       val_split=0.2,
       alb_transform=transforms,
       num_workers=4
   )
   
   # Model
   model = SegmentationModel(num_classes=3, learning_rate=1e-3)
   
   # Logger
   logger = MLFlowLogger(experiment_name="liver-segmentation")
   
   # Callbacks
   checkpoint_cb = MLFlowModelCheckpoint(
       monitor="val_dice",
       mode="max",
       register_model_name="liver-segmentation",
       register_model_on="train",
       additional_metadata={ # (Optional) Additional metadata necessary for deployment to cloud inference.
            "task_type": "semantic_segmentation",
            "labels": ["background", "liver", "tumor"],
            "need_gpu": False,                    # Whether GPU is required for inference
            "automatic_preprocessing": True       # Whether preprocessing is handled automatically
       }
   )
   
   # Trainer
   trainer = L.Trainer(
       max_epochs=100,
       logger=logger,
       callbacks=[checkpoint_cb],
       accelerator="gpu",
       devices=1,
       precision="16-mixed"
   )
   
   # Train
   trainer.fit(model, datamodule)

Model Management
----------------

Loading Trained Models
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import mlflow
   from datamint.mlflow.models import download_model_metadata
   
   # Load model from MLflow
   model_uri = "models:/liver-segmentation/1"
   model = mlflow.pytorch.load_model(model_uri)
   
   # Download additional metadata
   metadata = download_model_metadata(model_uri)
   print(f"Model architecture: {metadata.get('architecture')}")
   print(f"Training data: {metadata.get('data_modality')}")

Advanced Features
-----------------

Custom Collate Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from typing import List, Dict, Any
   
   def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
       """Custom collate function for variable-sized images."""
       
       # Handle images of different sizes
       images = [item['image'] for item in batch]
       max_h = max(img.shape[-2] for img in images)
       max_w = max(img.shape[-1] for img in images)
       
       # Pad images to same size
       padded_images = []
       for img in images:
           pad_h = max_h - img.shape[-2]
           pad_w = max_w - img.shape[-1]
           padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
           padded_images.append(padded)
       
       return {
           'image': torch.stack(padded_images),
           'metainfo': [item['metainfo'] for item in batch],
           'original_sizes': [(img.shape[-2], img.shape[-1]) for img in images]
       }
   
   # Use custom collate function
   dataloader = DataLoader(
       dataset,
       batch_size=8,
       collate_fn=custom_collate_fn
   )

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multi-GPU training setup
   trainer = L.Trainer(
       accelerator="gpu",
       devices=-1,  # Use all available GPUs
       strategy="ddp",  # Distributed Data Parallel
       precision="16-mixed",
       max_epochs=100,
       logger=logger,
       callbacks=[checkpoint_cb]
   )

Performance Optimization Tips
-----------------------------

1. **Data Loading Optimization**:

   .. code-block:: python
   
      # Use multiple workers for data loading
      datamodule = DatamintDataModule(
          project_name="large-dataset",
          batch_size=32,
          num_workers=8,  # Adjust based on CPU cores
          pin_memory=True  # For GPU training
      )

2. **Memory Management**:

   .. code-block:: python
   
      # For large datasets, use frame-by-frame loading
      dataset = Dataset(
          project_name="3d-medical-data",
          return_frame_by_frame=True,  # Process frames individually
          include_unannotated=False    # Reduce memory usage
      )

3. **Mixed Precision Training**:

   .. code-block:: python
   
      trainer = L.Trainer(
          precision="16-mixed",  # Use mixed precision
      )

4. **Gradient Accumulation**:

   .. code-block:: python
   
      trainer = L.Trainer(
          accumulate_grad_batches=4,  # Effective batch size = 4 * batch_size
      )

Advanced Dataset Customization
-------------------------------

For complex workflows, you can inherit from ``DatamintDataset`` to create custom dataset classes with specialized preprocessing and filtering logic.


Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**MLflow Connection Issues**:

- Verify Datamint API credentials
- Ensure proper project setup with ``datamint.mlflow.set_project()``