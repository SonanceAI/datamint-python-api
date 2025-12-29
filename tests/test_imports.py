"""
Test module to verify that all important datamint modules can be imported successfully.
This helps catch import issues early and ensures the package structure is correct.
"""
import pytest
import logging

# Set up logging to capture any import warnings
logging.basicConfig(level=logging.WARNING)
_LOGGER = logging.getLogger(__name__)


class TestImports:
    """Test suite for verifying module imports."""

    def test_main_package_import(self) -> None:
        """Test importing the main datamint package."""
        try:
            import datamint
            assert hasattr(datamint, '__version__')
            _LOGGER.info(f"Successfully imported datamint version: {datamint.__version__}")
        except ImportError as e:
            pytest.fail(f"Failed to import main datamint package: {e}")

    def test_dataset_imports(self) -> None:
        """Test importing dataset-related modules."""
        # Test direct import of Dataset class
        try:
            from datamint import Dataset
            assert Dataset is not None
            _LOGGER.info("Successfully imported Dataset from datamint")
        except ImportError as e:
            pytest.fail(f"Failed to import Dataset: {e}")

        # Test importing DatamintDataset directly
        try:
            from datamint.dataset.dataset import DatamintDataset
            assert DatamintDataset is not None
            _LOGGER.info("Successfully imported DatamintDataset")
        except ImportError as e:
            pytest.fail(f"Failed to import DatamintDataset: {e}")

        # Test importing base dataset
        try:
            from datamint.dataset.base_dataset import DatamintBaseDataset
            assert DatamintBaseDataset is not None
            _LOGGER.info("Successfully imported DatamintBaseDataset")
        except ImportError as e:
            pytest.fail(f"Failed to import DatamintBaseDataset: {e}")

    def test_api_imports(self) -> None:
        """Test importing API handler modules."""
        # Test direct import of APIHandler
        try:
            from datamint import Api
            assert Api is not None
            _LOGGER.info("Successfully imported APIHandler from datamint")
        except ImportError as e:
            pytest.fail(f"Failed to import APIHandler: {e}")

    def test_config_imports(self) -> None:
        """Test importing configuration modules."""
        try:
            from datamint import configs
            assert hasattr(configs, 'get_value')
            assert hasattr(configs, 'set_value')
            _LOGGER.info("Successfully imported configs module")
        except ImportError as e:
            pytest.fail(f"Failed to import configs: {e}")

    def test_examples_imports(self) -> None:
        """Test importing example modules."""
        try:
            from datamint.examples import ProjectMR
            assert ProjectMR is not None
            _LOGGER.info("Successfully imported examples")
        except ImportError as e:
            pytest.fail(f"Failed to import examples: {e}")

    def test_module_attributes(self) -> None:
        """Test that modules have expected attributes."""
        import datamint
        
        # Check main package attributes
        expected_attrs = ['__version__', '__name__']
        for attr in expected_attrs:
            assert hasattr(datamint, attr), f"Missing attribute {attr} in datamint package"
        
        # Test that classes have expected methods
        from datamint import Dataset
        expected_dataset_methods = ['__getitem__', '__len__']
        for method in expected_dataset_methods:
            assert hasattr(Dataset, method), f"Missing method {method} in Dataset class"

    def test_version_consistency(self) -> None:
        """Test that version information is consistent and accessible."""
        import datamint
        
        # Version should be a string
        assert isinstance(datamint.__version__, str)
        
        # Version should follow semantic versioning pattern (basic check)
        version_parts = datamint.__version__.split('.')
        assert len(version_parts) >= 2, f"Invalid version format: {datamint.__version__}"
        
        _LOGGER.info(f"Version consistency test passed: {datamint.__version__}")
