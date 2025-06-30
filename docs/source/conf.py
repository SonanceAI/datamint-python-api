# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata

project = 'DatamintAPI'

copyright = '2024, Sonance Team'
author = 'Sonance Team'

# The full version, including alpha/beta/rc tags
release = importlib.metadata.version("datamint")
master_doc = "index"
nitpicky = True
nitpick_ignore = {
    ("py:class", "pydicom.dataset.Dataset"),
    ("py:class", "PIL.Image.Image"),
    ("py:class", "np.ndarray"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "torch.nn.Module"),
    ("py:class", "cv2.VideoCapture"),
    ("py:class", "nibabel.filebasedimages.FileBasedImage"),
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.autosummary",
    'sphinx_tabs.tabs',
    'myst_parser',
    'sphinx_substitution_extensions',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
]

rst_prolog = """
.. |ExperimentClass| replace:: :py:class:`~datamint.experiment.experiment.Experiment`
.. |DatamintDatasetClass| replace:: :py:class:`~datamint.dataset.dataset.DatamintDataset`
.. |APIHandlerClass| replace:: :py:class:`~datamint.apihandler.api_handler.APIHandler`
"""

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = True

### sphinx_autodoc_typehints ###
typehints_fully_qualified = False
always_document_param_types = True
always_use_bars_union = True
##########
autodoc_inherit_docstrings = True

autosummary_imported_members = True  # Also documents imports in __init__.py

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "style_external_links": True,
    "logo_only": False,
}
html_static_path = ['_static']

# Add custom CSS to override theme defaults
html_css_files = [
    'custom.css',
]

html_favicon = "favicon.png"
