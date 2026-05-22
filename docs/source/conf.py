# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata

project = 'DatamintAPI'

copyright = '2024-2026, Sonance Team'
author = 'Sonance Team'

# The full version, including alpha/beta/rc tags
release = importlib.metadata.version("datamint")
master_doc = "index"
nitpicky = True
nitpick_ignore = {
    # Standard library types
    ("py:class", "typing.Literal"),
    ("py:class", "typing.Any"),
    ("py:class", "typing.Callable"),
    ("py:class", "typing.Iterable"),
    ("py:class", "typing.Iterator"),
    ("py:class", "typing.Mapping"),
    ("py:class", "typing.Generator"),
    ("py:class", "collections.abc.Sequence"),
    ("py:class", "collections.abc.Callable"),
    ("py:class", "collections.abc.Iterable"),
    ("py:class", "collections.abc.Iterator"),
    ("py:class", "collections.abc.Generator"),
    ("py:class", "pathlib.Path"),
    ("py:class", "datetime.date"),
    ("py:class", "datetime.datetime"),
    ("py:class", "abc.ABC"),
    ("py:class", "abc.ABCMeta"),
    ("py:data", "typing.Literal"),
    ("py:data", "typing.Any"),
    ("py:data", "typing.Callable"),
    ("py:data", "Ellipsis"),
    # Third-party types
    ("py:class", "pydicom.dataset.Dataset"),
    ("py:class", "PIL.Image.Image"),
    ("py:class", "Image.Image"),
    ("py:class", "np.ndarray"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "torch.nn.Module"),
    ("py:class", "torch.nn.modules.module.Module"),
    ("py:class", "torch.Tensor"),
    ("py:class", "torch.utils.data.DataLoader"),
    ("py:class", "torch.utils.data.dataloader.DataLoader"),
    ("py:class", "cv2.VideoCapture"),
    ("py:class", "nibabel.filebasedimages.FileBasedImage"),
    ("py:class", "nibabel.nifti1.Nifti1Image"),
    ("py:class", "pydantic.main.BaseModel"),
    ("py:class", "pydantic.config.ConfigDict"),
    ("py:class", "pydantic.functional_validators.BeforeValidator"),
    ("py:class", "pydantic.functional_serializers.PlainSerializer"),
    ("py:class", "pydantic.fields.FieldInfo"),
    ("py:class", "pydantic.types.PydanticUndefined"),
    ("py:class", "PydanticUndefined"),
    ("py:class", "httpx.HTTPStatusError"),
    ("py:class", "httpx.Response"),
    ("py:class", "httpx.Client"),
    ("py:class", "aiohttp.client_exceptions.ClientResponseError"),
    ("py:class", "aiohttp.client.ClientSession"),
    ("py:class", "lightning.pytorch.core.LightningDataModule"),
    ("py:class", "lightning.pytorch.trainer.trainer.Trainer"),
    ("py:class", "lightning.pytorch.callbacks.ModelCheckpoint"),
    ("py:class", "lightning.pytorch.core.module.LightningModule"),
    ("py:class", "lightning.pytorch.LightningModule"),
    ("py:class", "lightning.pytorch.core.LightningModule"),
    ("py:class", "lightning.pytorch.loggers.mlflow.MLFlowLogger"),
    ("py:class", "lightning.Trainer"),
    ("py:class", "lightning.LightningModule"),
    ("py:class", "albumentations.Compose"),
    ("py:class", "albumentations.Albumentation"),
    ("py:class", "albumentations.core.composition.BaseCompose"),
    ("py:class", "albumentations.core.transforms_interface.BasicTransform"),
    ("py:class", "albumentations.core.composition.BaseCompose"),
    ("py:class", "Path"),
    ("py:class", "io.IOBase"),
    ("py:class", "typing.IO"),
    # MLflow types
    ("py:class", "mlflow.data.dataset.Dataset"),
    ("py:class", "mlflow.data.dataset_source.DatasetSource"),
    ("py:class", "mlflow.pyfunc.model.PythonModel"),
    ("py:class", "mlflow.pyfunc.model.PythonModelContext"),
    ("py:class", "mlflow.pyfunc.PythonModel"),
    ("py:class", "mlflow.pyfunc.PyFuncModel"),
    ("py:class", "mlflow.models.signature.ModelSignature"),
    ("py:class", "mlflow.models.model.Model"),
    ("py:class", "mlflow.data.dataset.Dataset"),
    ("py:class", "mlflow.data.dataset_source.DatasetSource"),
    ("py:class", "mlflow.pytorch.model.PythonModel"),
    ("py:class", "mlflow.data.dataset.Dataset"),
    ("py:class", "mlflow.entities.Experiment"),
    ("py:class", "mlflow.entities.ExperimentTag"),
    ("py:class", "mlflow.types.Schema"),
    ("py:class", "pandas.core.frame.DataFrame"),
    # Torchmetrics
    ("py:class", "torchmetrics.Metric"),
    # Optimizer / scheduler types
    ("py:class", "torch.optim.lr_scheduler.ReduceLROnPlateau"),
    ("py:class", "torch.optim.LBFGS"),
    # Other
    ("py:class", "enum.StrEnum"),
    ("py:class", "datamint.entities.annotations.geometry._TwoPointGeometry"),
    ("py:class", "datamint.mlflow.lightning.callbacks.modelcheckpoint._BaseMLFlowModelCheckpoint"),
    ("py:class", "datamint.mlflow.lightning.callbacks.modelcheckpoint.MLflowModelCheckpoint"),
    ("py:class", "datamint.entities.annotations.annotation.AnnotationBase"),
    ("py:class", "datamint.entities.resource.BaseResource"),
    # TypeVar / generic references
    ("py:class", "datamint.entities.cache_manager.T"),
    # Internal classes not exposed in public API
    ("py:class", "datamint.entities.sliced_resource.SlicedVolumeResource"),
    ("py:class", "datamint.entities.annotations.annotation_spec.AnnotationSpec"),
    ("py:class", "datamint.entities.annotations.base_geometry.BaseGeometryAnnotation"),
    ("py:class", "datamint.entities.annotations.base_segmentation.BaseSegmentationAnnotation"),
    ("py:class", "datamint.lightning.datamodule.DatamintDataModule"),
    ("py:class", "datamint.lightning.trainers.lightning_modules.SegmentationModule"),
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.autosummary",
    'sphinx_design',
    'myst_parser',
    'sphinx_substitution_extensions',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.mermaid',
]

rst_prolog = """
.. |ApiClass| replace:: :py:class:`~datamint.api.client.Api`
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

# Ensure all modules are discoverable
autodoc_mock_imports = []

# Add type hints support
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
