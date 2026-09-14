"""Integration tests for the MLflow URL wiring in :class:`datamint.api.client.Api`.

These tests exercise ``Api.__init__`` directly with ``check_connection=False`` so
no network access is required; they assert that ``Api.mlflow_config`` receives the
URL derived by :func:`datamint.utils.urls.derive_mlflow_url` (with ``port`` left as
``None``) and that the API-URL fallback is used when derivation is impossible.
"""

import logging

import pytest

from datamint.api.client import Api

_API_KEY = "test-api-key"


def test_mlflow_config_derived_for_production_url(caplog: pytest.LogCaptureFixture) -> None:
    """Production API host derives the dedicated MLflow host on port 443."""
    with caplog.at_level(logging.INFO, logger="datamint.api.client"):
        api = Api(server_url="https://api.datamint.io", api_key=_API_KEY, check_connection=False)
    try:
        assert api.mlflow_config.server_url == "https://mlflow.datamint.io:443"
        assert api.mlflow_config.port is None
    finally:
        api.close()

    # Finding 4: the derived URL is logged at INFO so misconfiguration is visible.
    assert any(
        "https://mlflow.datamint.io:443" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    )


def test_mlflow_config_derived_for_localhost_url() -> None:
    """Non-production (localhost) keeps the API host on the MLflow dev port."""
    api = Api(server_url="http://localhost:3001", api_key=_API_KEY, check_connection=False)
    try:
        assert api.mlflow_config.server_url == "http://localhost:5000"
        assert api.mlflow_config.port is None
    finally:
        api.close()


def test_mlflow_config_falls_back_and_warns_without_hostname(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scheme-less/invalid URL cannot be derived from, so the API URL is reused."""
    with caplog.at_level(logging.WARNING, logger="datamint.api.client"):
        api = Api(server_url="not-a-url", api_key=_API_KEY, check_connection=False)
    try:
        assert api.mlflow_config.server_url == "not-a-url"
        assert api.mlflow_config.port is None
    finally:
        api.close()

    assert any(
        "Could not derive MLflow server URL" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )