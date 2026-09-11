"""Unit tests for datamint.utils.urls.derive_mlflow_url."""

import pytest

from datamint.utils.urls import derive_mlflow_url


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        # Production: api.<domain> -> mlflow.<domain> over HTTPS on port 443.
        ("https://api.datamint.io", "https://mlflow.datamint.io:443"),
        ("http://api.datamint.io", "https://mlflow.datamint.io:443"),
        ("https://api.datamint.io/", "https://mlflow.datamint.io:443"),
        ("https://api.datamint.io:8080", "https://mlflow.datamint.io:443"),
        ("https://api.staging.datamint.io", "https://mlflow.staging.datamint.io:443"),
        # Hostname case is normalized by urlparse (lowercased).
        ("https://API.Datamint.IO", "https://mlflow.datamint.io:443"),
        # Path components are ignored; only the hostname matters.
        ("https://api.datamint.io/v1/", "https://mlflow.datamint.io:443"),
        # Locks in the literal-spec behavior: any hostname starting with "api."
        # has the prefix swapped for "mlflow.", even for non-datamint domains.
        ("https://api.foo.example.com", "https://mlflow.foo.example.com:443"),
        # Localhost/dev: keep the same host on HTTP port 5000.
        ("http://localhost:3001", "http://localhost:5000"),
        ("http://localhost:3001/", "http://localhost:5000"),
        ("http://localhost", "http://localhost:5000"),
        ("http://127.0.0.1:3001", "http://127.0.0.1:5000"),
        ("https://dev.internal.example.com", "http://dev.internal.example.com:5000"),
        # IPv6 literals must keep their brackets in the derived URL.
        ("http://[::1]:3001", "http://[::1]:5000"),
    ],
)
def test_derive_mlflow_url(api_url: str, expected: str) -> None:
    assert derive_mlflow_url(api_url) == expected


@pytest.mark.parametrize("api_url", [None, "", "not-a-url", "://missing-host"])
def test_derive_mlflow_url_returns_none_without_hostname(api_url: str | None) -> None:
    assert derive_mlflow_url(api_url) is None
