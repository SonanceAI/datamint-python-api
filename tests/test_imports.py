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

    def test_api_handler_imports(self) -> None:
        """Test importing API handler modules."""
        # Test direct import of APIHandler
        try:
            from datamint import APIHandler
            assert APIHandler is not None
            _LOGGER.info("Successfully imported APIHandler from datamint")
        except ImportError as e:
            pytest.fail(f"Failed to import APIHandler: {e}")

        # Test importing APIHandler directly
        try:
            from datamint.apihandler.api_handler import APIHandler as APIHandlerClass
            assert APIHandlerClass is not None
            _LOGGER.info("Successfully imported APIHandler class directly")
        except ImportError as e:
            pytest.fail(f"Failed to import APIHandler class directly: {e}")

        # Test importing base API handler
        try:
            from datamint.apihandler.base_api_handler import DatamintException
            assert DatamintException is not None
            _LOGGER.info("Successfully imported DatamintException")
        except ImportError as e:
            pytest.fail(f"Failed to import DatamintException: {e}")

    def test_utils_imports(self) -> None:
        """Test importing utility modules."""
        # Test logging utils
        try:
            from datamint.utils.logging_utils import load_cmdline_logging_config
            assert load_cmdline_logging_config is not None
            _LOGGER.info("Successfully imported logging_utils")
        except ImportError as e:
            pytest.fail(f"Failed to import logging_utils: {e}")

        # Test IO utils
        try:
            from medimgkit.readers import read_array_normalized
            assert read_array_normalized is not None
            _LOGGER.info("Successfully imported io_utils")
        except ImportError as e:
            pytest.fail(f"Failed to import io_utils: {e}")

        # Test DICOM utils (if available)
        try:
            from medimgkit.dicom_utils import is_dicom
            assert is_dicom is not None
            _LOGGER.info("Successfully imported dicom_utils")
        except ImportError as e:
            pytest.fail(f"Failed to import dicom_utils: {e}")

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
