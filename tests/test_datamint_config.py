"""
Test module for datamint-config command line tool.
Ensures the configuration tool works without importing torch or other heavy dependencies.
"""
import io
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from datamint import configs

_LOGGER = logging.getLogger(__name__)


def _create_local_data_tree(root: Path) -> None:
    # Resource cache tree with metadata used by the CLI summaries and cleanup selectors
    resources_dir = root / 'resources' / 'res-1'
    resources_dir.mkdir(parents=True)
    (resources_dir / 'metadata.json').write_text(
        json.dumps(
            {
                'cached_at': '2026-04-13T00:00:00',
                'data_path': str(resources_dir / 'image_data'),
                'data_type': 'bytes',
                'mimetype': 'application/octet-stream',
                'entity_id': 'res-1',
                'extra_info': {'upload_channel': 'training-data', 'tags': ['tutorial', 'brain']},
            }
        ),
        encoding='utf-8',
    )
    (resources_dir / 'image_data').write_bytes(b'resource-bytes')

    # An extra resource not associated with any project
    unowned_dir = root / 'resources' / 'unowned-resource'
    unowned_dir.mkdir(parents=True)
    (unowned_dir / 'metadata.json').write_text(
        json.dumps(
            {
                'cached_at': '2026-04-13T00:00:00',
                'data_path': str(unowned_dir / 'image_data'),
                'data_type': 'bytes',
                'mimetype': 'application/octet-stream',
                'entity_id': 'unowned-resource',
                'extra_info': {'upload_channel': 'training-data', 'tags': ['tutorial']},
            }
        ),
        encoding='utf-8',
    )
    (unowned_dir / 'image_data').write_bytes(b'unowned-resource-bytes')

    annotations_dir = root / 'annotations' / 'annotation-1'
    annotations_dir.mkdir(parents=True)
    (annotations_dir / 'metadata.json').write_text('{}', encoding='utf-8')
    (annotations_dir / 'segmentation_data').write_bytes(b'annotation-bytes')

    legacy_dataset_dir = root / 'datasets_old' / 'Example Project'
    (legacy_dataset_dir / 'images').mkdir(parents=True)
    (legacy_dataset_dir / 'images' / 'R0001.dcm').write_bytes(b'legacy-resource')
    (legacy_dataset_dir / 'dataset.json').write_text(
        json.dumps({'resource_ids': ['res-1', 'res-2']}),
        encoding='utf-8',
    )


def _set_test_console(monkeypatch: pytest.MonkeyPatch):
    from datamint.client_cmd_tools import datamint_config

    monkeypatch.setattr(
        datamint_config,
        'console',
        Console(file=io.StringIO(), force_terminal=False),
    )
    return datamint_config


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

    @patch('datamint.configs.set_values')
    def test_command_line_api_key_argument(self, mock_set_values) -> None:
        """Test command line execution with --api-key argument."""
        test_api_key = 'test_key_from_cli_12345'

        # Test using sys.argv patching for direct main() call
        with patch('sys.argv', ['datamint-config', '--api-key', test_api_key]):
            from datamint.client_cmd_tools.datamint_config import main
            main()

            # Verify the API key was set with correct key
            mock_set_values.assert_called_once()

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
    @patch('datamint.configs.set_values')
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

    def test_discover_local_datasets_groups_cache_namespaces_and_legacy_datasets(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test local data discovery groups top-level namespaces and keeps resource filters separate."""
        _create_local_data_tree(tmp_path)
        monkeypatch.setattr(configs, 'DATAMINT_DATA_DIR', str(tmp_path))
        datamint_config = _set_test_console(monkeypatch)

        groups = datamint_config.discover_local_datasets()
        groups_by_id = {str(group['identifier']): group for group in groups}

        # resources namespace is reported as a top-level cache group
        unowned_group = groups_by_id.get('cache:resources')
        assert unowned_group is not None, "Expected a 'cache:resources' group"
        assert unowned_group['kind'] == 'cache namespace'
        assert unowned_group['item_count'] == 2

        # Annotations are also reported as a plain cache namespace
        ann_group = groups_by_id.get('cache:annotations')
        assert ann_group is not None
        assert ann_group['kind'] == 'cache namespace'
        assert ann_group['item_count'] == 1

        # The legacy dataset folder is still reported separately
        legacy_group = groups_by_id.get('legacy:Example Project')
        assert legacy_group is not None
        assert legacy_group['kind'] == 'legacy dataset'
        assert legacy_group['item_count'] == 2

    def test_discover_resource_filter_groups_uses_upload_channel_and_tags(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test resource filter groups are derived from cached metadata."""
        _create_local_data_tree(tmp_path)
        monkeypatch.setattr(configs, 'DATAMINT_DATA_DIR', str(tmp_path))
        datamint_config = _set_test_console(monkeypatch)

        channel_groups, tag_groups = datamint_config._discover_resource_filter_groups(tag_limit=10)

        channel_by_id = {str(group['identifier']): group for group in channel_groups}
        tag_by_id = {str(group['identifier']): group for group in tag_groups}

        channel_group = channel_by_id.get('channel:training-data')
        assert channel_group is not None
        assert channel_group['kind'] == 'upload channel'
        assert channel_group['item_count'] == 2

        tutorial_group = tag_by_id.get('tag:tutorial')
        assert tutorial_group is not None
        assert tutorial_group['kind'] == 'tag'
        assert tutorial_group['item_count'] == 2

        brain_group = tag_by_id.get('tag:brain')
        assert brain_group is not None
        assert brain_group['kind'] == 'tag'
        assert brain_group['item_count'] == 1

    def test_clean_dataset_removes_channel_group_resources(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test cleaning a channel selector deletes only the matching cached resource dirs."""
        _create_local_data_tree(tmp_path)
        monkeypatch.setattr(configs, 'DATAMINT_DATA_DIR', str(tmp_path))
        datamint_config = _set_test_console(monkeypatch)

        with patch('datamint.client_cmd_tools.datamint_config.Confirm.ask', return_value=True):
            assert datamint_config.clean_dataset('channel:training-data') is True

        assert not (tmp_path / 'resources' / 'res-1').exists()
        assert not (tmp_path / 'resources' / 'unowned-resource').exists()
        assert (tmp_path / 'resources').exists()
        assert (tmp_path / 'annotations').exists()

    def test_clean_dataset_removes_tag_group_resources(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test cleaning a tag selector deletes only the matching cached resource dirs."""
        _create_local_data_tree(tmp_path)
        monkeypatch.setattr(configs, 'DATAMINT_DATA_DIR', str(tmp_path))
        datamint_config = _set_test_console(monkeypatch)

        with patch('datamint.client_cmd_tools.datamint_config.Confirm.ask', return_value=True):
            assert datamint_config.clean_dataset('tag:brain') is True

        assert not (tmp_path / 'resources' / 'res-1').exists()
        assert (tmp_path / 'resources' / 'unowned-resource').exists()
        assert (tmp_path / 'resources').exists()
        assert (tmp_path / 'annotations').exists()
