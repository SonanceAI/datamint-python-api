"""
Datamint API is a Python package that provides a simple interface to the Datamint API.

TODO...
"""

from .dataset.dataset import DatamintDataset as Dataset
from .api_handler import APIHandler
from .experiment import Experiment
import importlib.metadata

__name__ = "datamintapi"
__version__ = importlib.metadata.version(__name__)
