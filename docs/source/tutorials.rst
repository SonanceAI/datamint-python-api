Tutorials
=========

The notebooks below are available in the `notebooks/ directory <https://github.com/SonanceAI/datamint-python-api/tree/main/notebooks>`_
of the GitHub repository. Run them locally to learn how to use the Datamint Python API across different scenarios.

Getting Started
---------------

* `01_upload_data.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/01_getting_started/01_upload_data.ipynb>`_: Upload images, DICOMs, and other resources to a Datamint project.
* `02_explore_data.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/01_getting_started/02_explore_data.ipynb>`_: Query and explore resources already in a project.

Annotations
-----------

* `01_upload_annotations.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/02_annotations/01_upload_annotations.ipynb>`_: Import and manage image-level and frame-level classification annotations.
* `02_geometry_annotations.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/02_annotations/02_geometry_annotations.ipynb>`_: Upload bounding boxes, lines, and other geometry annotations.

Datasets
--------

* `01_project_scoped_splits.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/03_datasets/01_project_scoped_splits.ipynb>`_: Assign project-scoped train/val/test splits, inspect split records, and replay historical snapshots.
* `02_patient_wise_splits.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/03_datasets/02_patient_wise_splits.ipynb>`_: Split datasets by patient to avoid data leakage between train and test sets.
* `03_build_dataset.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/03_datasets/03_build_dataset.ipynb>`_: Build and configure a PyTorch dataset from a Datamint project.
* `04_volume_dataset.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/03_datasets/04_volume_dataset.ipynb>`_: Work with 3D volume datasets (NIfTI, DICOM series).

Experiment Tracking
-------------------

* `01_mlflow_manual_logging.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/04_experiment_tracking/01_mlflow_manual_logging.ipynb>`_: Log models and experiments manually to MLflow via Datamint.

Deployment
----------

* `01_deploy_registered_model.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/05_deployment/01_deploy_registered_model.ipynb>`_: Deploy a model registered in MLflow as a managed Datamint endpoint.
* `02_deploy_external_model.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/05_deployment/02_deploy_external_model.ipynb>`_: Adapt and deploy an externally-trained model in Datamint.
* `03_validate_model.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/05_deployment/03_validate_model.ipynb>`_: Validate a model before promoting it to production.

End-to-End Examples
-------------------

Complete workflows from data upload to deployment.

**Slice-based (2D)**

* `01_fracatlas_classification.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/06_end_to_end/slice_based/01_fracatlas_classification.ipynb>`_: End-to-end classification pipeline on the FracAtlas dataset.
* `02_busi_segmentation.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/06_end_to_end/slice_based/02_busi_segmentation.ipynb>`_: Train a 2D segmentation model on the BUSI dataset with ``UNetPPTrainer``, including custom model integration.
* `03_bccd_detection.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/06_end_to_end/slice_based/03_bccd_detection.ipynb>`_: Object detection pipeline on the BCCD blood cell dataset.

**Full 3D**

* `01_synapse_unetrpp.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/06_end_to_end/full_3d/01_synapse_unetrpp.ipynb>`_: Volumetric segmentation on the Synapse dataset using UNETR++.
* `02_synapse_nnunet.ipynb <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/06_end_to_end/full_3d/02_synapse_nnunet.ipynb>`_: Volumetric segmentation on the Synapse dataset using nnUNet.
