"""
Datamint API package alias.

This module serves as an alias for the datamintapi package.
"""

from datamintapi import *
import importlib.metadata

__name__ = "datamint"
__version__ = importlib.metadata.version(__name__)