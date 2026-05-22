MLflow Integration
==================

The ``datamint.mlflow`` module provides seamless MLflow integration for experiment
tracking, model logging, and model deployment.

.. toctree::
   :maxdepth: 2
   :caption: MLflow Components

   datamint.mlflow.model
   datamint.mlflow.dataset
   datamint.mlflow.checkpointing

Overview
--------

Key features:

- **Automatic Experiment Tracking**: Experiments are automatically associated with Datamint projects
- **Model Registration**: Trained models are registered in MLflow and can be deployed to Datamint
- **Dataset Versioning**: Datasets are logged as MLflow artifacts with metadata
- **Custom Checkpointing**: MLflowModelCheckpoint integrates with Lightning callbacks
- **Flavor Support**: Custom Datamint flavors for segmentation and classification models

Automatic Configuration
-----------------------

The MLflow module auto-configures itself on first import:

.. code-block:: python

    # This automatically sets up MLflow environment
    import datamint.mlflow

    # Or explicitly:
    from datamint.mlflow import ensure_mlflow_configured
    ensure_mlflow_configured()

Environment Setup
-----------------

.. automodule:: datamint.mlflow.env_utils
   :members: setup_mlflow_environment, ensure_mlflow_configured
   :undoc-members:

.. automodule:: datamint.mlflow.env_vars
   :members:
   :undoc-members:

MLflow Dataset
--------------

.. automodule:: datamint.mlflow.data.datamint_dataset
   :members: DatamintMLflowDataset
   :undoc-members:
   :show-inheritance:

The :py:class:`~datamint.mlflow.data.datamint_dataset.DatamintMLflowDataset` wraps
a :py:class:`~datamint.dataset.base.DatamintBaseDataset` for MLflow artifact logging.

Model Flavors
-------------

.. automodule:: datamint.mlflow.flavors.model
   :members: BaseDatamintModel, DatamintModel
   :undoc-members:
   :show-inheritance:

.. automodule:: datamint.mlflow.flavors.task_type
   :members:
   :undoc-members:

.. automodule:: datamint.mlflow.flavors.prediction_router
   :members:
   :undoc-members:

.. automodule:: datamint.mlflow.flavors.prediction_modes
   :members:
   :undoc-members:

.. automodule:: datamint.mlflow.flavors.datamint_flavor
   :members:
   :undoc-members:

Checkpointing
-------------

.. automodule:: datamint.mlflow.lightning.callbacks.modelcheckpoint
   :members: MLflowModelCheckpoint
   :undoc-members:
   :show-inheritance:

The :py:class:`~datamint.mlflow.lightning.callbacks.modelcheckpoint.MLflowModelCheckpoint`
extends Lightning's ``ModelCheckpoint`` to automatically log checkpoints as MLflow artifacts.

MLflow Tracking
---------------

.. automodule:: datamint.mlflow.tracking.fluent
   :members: set_project
   :undoc-members:

.. automodule:: datamint.mlflow.tracking.default_experiment
   :members:
   :undoc-members:

.. automodule:: datamint.mlflow.tracking.datamint_store
   :members:
   :undoc-members:

Artifact Repository
-------------------

.. automodule:: datamint.mlflow.artifact.datamint_artifacts_repo
   :members:
   :undoc-members:

Usage Example
-------------

.. code-block:: python

    import lightning as L
    from datamint.dataset import ImageDataset
    from datamint.lightning import DatamintDataModule, UNetPPTrainer

    # Create dataset and datamodule
    dataset = ImageDataset(project="Liver Segmentation")
    datamodule = DatamintDataModule(
        dataset,
        batch_size=16,
        split={'train': 0.8, 'val': 0.1, 'test': 0.1},
    )

    # Use trainer with automatic MLflow integration
    trainer = UNetPPTrainer(
        project="Liver Segmentation",
        image_size=256,
        batch_size=16,
        max_epochs=50,
        accelerator="gpu",
        register_model=True,  # Auto-register in MLflow
    )

    results = trainer.fit()

    # Model is automatically logged and registered
    print(results["test_results"])

    import mlflow
    mlflow.get_experiment_by_name("Liver Segmentation")  # Verify experiment exists
    mlflow.search_runs(experiment_names=["Liver Segmentation"])  # View runs