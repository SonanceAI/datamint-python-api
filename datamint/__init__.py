"""
Datamint API package alias.
"""

import importlib.metadata
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .api.client import Api
    # New modular datasets
    from .dataset.image_dataset import ImageDataset
    from .dataset.volume_dataset import VolumeDataset
    from .mlflow.flavors.validation import validate_model, ValidationReport, ValidationIssue, ModelValidationError

else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=['dataset', "dataset.dataset", "examples"],
        submod_attrs={
            "api.client": ["Api"],
            # New modular dataset classes
            "dataset.image_dataset": ["ImageDataset"],
            "dataset.volume_dataset": ["VolumeDataset"],
            "mlflow.flavors.validation": ["validate_model", "ValidationReport",
                                          "ValidationIssue", "ModelValidationError"],
        },
    )

__name__ = "datamint"
__version__ = importlib.metadata.version(__name__)