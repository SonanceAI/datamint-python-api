"""Shared MLflow tag keys used by Datamint's model registry integration."""

DATAMINT_LOGGED_MODEL_ID_TAG = 'datamint.logged_model_id'
"""ModelVersion tag holding the LoggedModel.model_id it was registered from.

Stamped at registration time in ``_BaseMLFlowModelCheckpoint.register_model()``,
read back by ``ModelVersion.get_metrics()`` to find the LoggedModel that carries
the training/test metrics for this version.
"""
