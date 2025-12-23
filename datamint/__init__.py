"""
Datamint API package alias.
"""

import importlib.metadata
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .dataset.dataset import DatamintDataset as Dataset
    from .apihandler.api_handler import APIHandler
    from .api.client import Api
else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=['dataset', "dataset.dataset", "apihandler.api_handler"],
        submod_attrs={
            "dataset.dataset": ["DatamintDataset"],
            "dataset": ['Dataset'],
            "apihandler.api_handler": ["APIHandler"],
            "api.client": ["Api"],
        },
    )

__name__ = "datamint"
__version__ = importlib.metadata.version(__name__)