"""Combined LightningModule + BaseDatamintModel base for built-in trainers."""
from __future__ import annotations

import lightning as L

from datamint.mlflow.flavors.model import BaseDatamintModel, ModelSettings
from mlflow.pyfunc.model import PythonModelContext


class DatamintLightningModule(L.LightningModule, BaseDatamintModel):
    """A :class:`~lightning.LightningModule` that is also a
    :class:`~datamint.mlflow.flavors.model.BaseDatamintModel`.

    Built-in trainers use this as the base for their default models so that
    the trained module can be logged once with ``datamint_flavor`` — no
    separate adapter step is required.
    """

    def __init__(self, settings: ModelSettings | None = None) -> None:
        L.LightningModule.__init__(self)
        BaseDatamintModel.__init__(self, settings=settings)

    # ------------------------------------------------------------------
    # MLflow lifecycle
    # ------------------------------------------------------------------

    def load_context(self, context: PythonModelContext) -> None:
        """Move weights to the configured device and set eval mode on MLflow load."""
        device = (context.model_config or {}).get('device', 'cpu')
        self.to(device)
        self.eval()
