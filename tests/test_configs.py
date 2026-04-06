from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from datamint import configs


@pytest.fixture
def isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the config module to a temporary config file."""
    config_dir = tmp_path / "config"
    config_file = config_dir / "datamintapi.yaml"

    monkeypatch.setattr(configs, "DIRS", SimpleNamespace(user_config_dir=str(config_dir)))
    monkeypatch.setattr(configs, "CONFIG_FILE", str(config_file))

    return config_file


def test_read_config_returns_defaults_for_empty_file(isolated_config: Path) -> None:
    isolated_config.parent.mkdir()
    isolated_config.write_text("", encoding="utf-8")

    assert configs.read_config() == configs.DEFAULT_VALUES


def test_set_values_persists_data_with_default_url(isolated_config: Path) -> None:
    configs.set_values({configs.APIKEY_KEY: "saved-key"})

    assert isolated_config.exists()

    stored_config = yaml.safe_load(isolated_config.read_text(encoding="utf-8"))
    assert stored_config[configs.APIKEY_KEY] == "saved-key"
    assert stored_config[configs.APIURL_KEY] == configs.DEFAULT_VALUES[configs.APIURL_KEY]


def test_get_value_prefers_environment_variable(isolated_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolated_config.parent.mkdir()
    isolated_config.write_text(
        yaml.safe_dump({configs.APIKEY_KEY: "saved-key"}),
        encoding="utf-8",
    )
    monkeypatch.setenv(configs.get_env_var_name(configs.APIKEY_KEY), "env-key")

    assert configs.get_value(configs.APIKEY_KEY) == "env-key"
    assert configs.get_value(configs.APIKEY_KEY, include_envvars=False) == "saved-key"