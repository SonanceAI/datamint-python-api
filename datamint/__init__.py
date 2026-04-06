"""
Datamint API package alias.
"""

import importlib.metadata
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Legacy
    from .dataset.dataset import DatamintDataset as Dataset
    
    from .api.client import Api
    # New modular datasets
    from .dataset.image_dataset import ImageDataset
    from .dataset.volume_dataset import VolumeDataset
    
else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=['dataset', "dataset.dataset"],
        submod_attrs={
            # Legacy exports
            "dataset.dataset": ["DatamintDataset"],
            "dataset": ['Dataset'],
            "api.client": ["Api"],
            # New modular dataset classes
            "dataset.image_dataset": ["ImageDataset"],
            "dataset.volume_dataset": ["VolumeDataset"],
        },
    )

__name__ = "datamint"
__version__ = importlib.metadata.version(__name__)