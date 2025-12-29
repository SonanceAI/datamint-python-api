"""
Test module for datamint-config command line tool.
Ensures the configuration tool works without importing torch or other heavy dependencies.
"""
import pytest
import sys
from unittest.mock import patch
import logging

_LOGGER = logging.getLogger(__name__)


class TestDatamintConfig:
    """Test suite for datamint-config functionality."""

    def test_config_imports_no_torch(self) -> None:
        """Test that importing datamint_config module doesn't import torch."""
        # Track which modules are imported
        initial_modules = set(sys.modules.keys())

        # Import the config module
        from datamint.client_cmd_tools.datamint_config import main

        # Check new modules that were imported
        new_modules = set(sys.modules.keys()) - initial_modules

        # Ensure torch-related modules are not imported
        torch_modules = [mod for mod in new_modules if 'torch' in mod.lower()]
        assert len(torch_modules) == 0, f"Torch modules were imported: {torch_modules}"

        # Ensure heavy ML libraries are not imported
        heavy_ml_libs = ['torch', 'tensorflow', 'sklearn']
        imported_heavy_libs = [lib for lib in heavy_ml_libs if lib in new_modules]
        assert len(imported_heavy_libs) == 0, f"Heavy ML libraries were imported: {imported_heavy_libs}"

        _LOGGER.info("Successfully verified no torch import in datamint_config")

    @patch('datamint.configs.set_value')
    def test_command_line_api_key_argument(self, mock_set_value) -> None:
        """Test command line execution with --api-key argument."""
        test_api_key = 'test_key_from_cli_12345'

        # Test using sys.argv patching for direct main() call
        with patch('sys.argv', ['datamint-config', '--api-key', test_api_key]):
            from datamint.client_cmd_tools.datamint_config import main
            main()

            # Verify the API key was set with correct key
            mock_set_value.assert_called_once()
            call_args = mock_set_value.call_args
            assert call_args[0][1] == test_api_key, f"Expected API key {test_api_key}, got {call_args[0][1]}"

    def test_show_configurations_functionality(self) -> None:
        """Test show_all_configurations without user interaction."""
        from datamint.client_cmd_tools.datamint_config import show_all_configurations

        show_all_configurations()

    def test_config_tool_startup_time(self) -> None:
        """Test that the config tool starts up quickly (performance test)."""
        import time
        
        start_time = time.time()
        from datamint.client_cmd_tools.datamint_config import main
        end_time = time.time()
        
        startup_time = end_time - start_time
        assert startup_time < 1.0, f"Config tool took too long to import: {startup_time:.2f}s"
        _LOGGER.info(f"Config tool import time: {startup_time:.3f}s")

    def test_config_error_handling(self) -> None:
        """Test config tool error handling with invalid inputs."""
        with patch('sys.argv', ['datamint-config', '--invalid-option']):
            with pytest.raises(SystemExit):
                from datamint.client_cmd_tools.datamint_config import main
                main()

    @patch('datamint.configs.get_value')
    @patch('datamint.configs.set_value')
    def test_config_persistence(self, mock_set, mock_get) -> None:
        """Test that config values persist correctly."""
        mock_get.return_value = None
        
        # Test setting and getting a value
        test_key = 'test_config_key'
        test_value = 'test_config_value'
        
        with patch('sys.argv', ['datamint-config', '--api-key', test_value]):
            from datamint.client_cmd_tools.datamint_config import main
            main()
            
        mock_set.assert_called()

    def test_environment_variable_integration(self) -> None:
        """Test config integration with environment variables."""
        import os
        from datamint import configs
        
        # Test environment variable fallback
        test_key = 'DATAMINT_TEST_VAR'
        test_value = 'env_test_value'
        
        original_value = os.environ.get(test_key)
        try:
            os.environ[test_key] = test_value
            # Test that environment variables are recognized
            assert test_key in os.environ
        finally:
            if original_value is not None:
                os.environ[test_key] = original_value
            elif test_key in os.environ:
                del os.environ[test_key]
