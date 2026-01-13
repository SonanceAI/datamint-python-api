"""
Test that MLflow environment is properly configured regardless of import order.
"""

import pytest
from unittest.mock import patch
import sys
import os
from itertools import permutations


# Test configuration values
TEST_API_URL = "http://localhost:3001"
TEST_API_KEY = "test-api-key-12345"
TEST_MLFLOW_URI = "http://localhost:5000"


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables and modules before and after each test."""
    # Save original environment variables
    original_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    original_tracking_token = os.environ.get('MLFLOW_TRACKING_TOKEN')
    
    # Clean up before test
    if 'MLFLOW_TRACKING_URI' in os.environ:
        del os.environ['MLFLOW_TRACKING_URI']
    if 'MLFLOW_TRACKING_TOKEN' in os.environ:
        del os.environ['MLFLOW_TRACKING_TOKEN']
    
    yield
    
    # Restore original environment variables after test
    if original_tracking_uri is not None:
        os.environ['MLFLOW_TRACKING_URI'] = original_tracking_uri
    elif 'MLFLOW_TRACKING_URI' in os.environ:
        del os.environ['MLFLOW_TRACKING_URI']
    
    if original_tracking_token is not None:
        os.environ['MLFLOW_TRACKING_TOKEN'] = original_tracking_token
    elif 'MLFLOW_TRACKING_TOKEN' in os.environ:
        del os.environ['MLFLOW_TRACKING_TOKEN']


@pytest.fixture(autouse=True)
def mock_datamint_config():
    """Mock Datamint configuration to return test values."""
    with patch('datamint.mlflow.env_utils.get_datamint_api_url', return_value=TEST_API_URL), \
         patch('datamint.mlflow.env_utils.get_datamint_api_key', return_value=TEST_API_KEY):
        yield


# Store the expected tracking URI based on Datamint config
def get_expected_tracking_uri():
    """Get the expected MLflow tracking URI from Datamint configuration."""
    from datamint.mlflow.env_utils import _get_mlflowdatamint_uri
    return _get_mlflowdatamint_uri()


# Define all the imports we want to test
IMPORT_STATEMENTS = [
    ("MLFlowModelCheckpoint", "from datamint.mlflow.lightning.callbacks import MLFlowModelCheckpoint"),
    ("MLFlowLogger", "from lightning.pytorch.loggers import MLFlowLogger"),
    ("set_project", "from datamint.mlflow import set_project"),
]


def cleanup_modules():
    """Clean up imported modules to ensure fresh imports."""
    modules_to_remove = [
        'datamint.mlflow',
        'datamint.mlflow.lightning',
        'datamint.mlflow.lightning.callbacks',
        'lightning.pytorch.loggers',
        'lightning.pytorch.loggers.mlflow',
        'mlflow',
        'mlflow.tracking',
        'mlflow.tracking._tracking_service',
        'mlflow.tracking._tracking_service.utils',
    ]
    
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]


@pytest.mark.parametrize("import_order", list(permutations(range(3))))
def test_mlflow_uri_patching_all_import_orders(import_order, mock_datamint_config):
    """
    Test that MLFlowLogger gets the patched URI regardless of import order.
    
    This tests all 6 permutations of importing:
    - from datamint.mlflow.lightning.callbacks import MLFlowModelCheckpoint
    - from lightning.pytorch.loggers import MLFlowLogger
    - from datamint.mlflow import set_project
    """
    # Clean up modules before test
    cleanup_modules()
    
    # Expected URI should be the test MLflow URI
    expected_uri = TEST_MLFLOW_URI
    
    # Execute imports in the specified order
    namespace = {}
    for idx in import_order:
        name, import_stmt = IMPORT_STATEMENTS[idx]
        exec(import_stmt, namespace)
    
    # Get MLFlowLogger from namespace (it should have been imported)
    MLFlowLogger = namespace.get('MLFlowLogger')
    # Create an MLFlowLogger instance
    mllogger = MLFlowLogger(experiment_name="expname", run_name="runname")
    
    # Verify the tracking URI is the patched one
    assert mllogger._tracking_uri == expected_uri, (
        f"Import order {import_order} failed: "
        f"expected URI '{expected_uri}', got '{mllogger._tracking_uri}'"
    )

    import mlflow
    assert mlflow.get_tracking_uri() == expected_uri



def test_mlflow_environment_variables_set(mock_datamint_config):
    """
    Test that MLflow environment variables are properly set.
    """
    from datamint.mlflow.env_utils import setup_mlflow_environment
    
    # Setup environment
    success = setup_mlflow_environment(overwrite=True, set_mlflow=False)
    
    assert success, "setup_mlflow_environment should succeed with mocked config"
    
    # Check environment variables
    assert os.getenv('MLFLOW_TRACKING_URI') is not None, "MLFLOW_TRACKING_URI not set"
    assert os.getenv('MLFLOW_TRACKING_TOKEN') is not None, "MLFLOW_TRACKING_TOKEN not set"
    
    assert os.getenv('MLFLOW_TRACKING_URI') == TEST_MLFLOW_URI, (
        f"Expected URI '{TEST_MLFLOW_URI}', got '{os.getenv('MLFLOW_TRACKING_URI')}'"
    )
    assert os.getenv('MLFLOW_TRACKING_TOKEN') == TEST_API_KEY, (
        f"Expected token '{TEST_API_KEY}', got '{os.getenv('MLFLOW_TRACKING_TOKEN')}'"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
