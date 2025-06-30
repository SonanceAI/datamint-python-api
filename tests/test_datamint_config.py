"""
Test module for datamint-config command line tool.
Ensures the configuration tool works without importing torch or other heavy dependencies.
"""
import pytest
import sys
import subprocess
import tempfile
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
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

        # Import and initialize the config module
        from datamint.client_cmd_tools.datamint_config import main

        duration = time.time() - start_time

        # Config tool should start very quickly (under 1 second)
        assert duration < 0.5, f"Config tool startup too slow: {duration:.3f} seconds"
        _LOGGER.info(f"Config tool startup time: {duration:.3f} seconds")
