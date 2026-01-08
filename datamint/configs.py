import yaml
import os
import logging
from platformdirs import PlatformDirs
from typing import Any

APIURL_KEY = 'default_api_url'
APIKEY_KEY = 'api_key'

ENV_VARS = {
    APIKEY_KEY: 'DATAMINT_API_KEY',
    APIURL_KEY: 'DATAMINT_API_URL'
}

DEFAULT_VALUES = {
    APIURL_KEY: 'https://api.datamint.io'
}

_LOGGER = logging.getLogger(__name__)

DIRS = PlatformDirs(appname='datamintapi')
CONFIG_FILE = os.path.join(DIRS.user_config_dir, 'datamintapi.yaml')
try:
    DATAMINT_DATA_DIR = os.path.join(os.path.expanduser("~"), '.datamint')
except Exception as e:
    _LOGGER.error(f"Could not determine home directory: {e}")
    DATAMINT_DATA_DIR = None


def get_env_var_name(key: str) -> str:
    return ENV_VARS[key]


def read_config() -> dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as configfile:
            config = yaml.safe_load(configfile)
        config.update({k: v for k, v in DEFAULT_VALUES.items() if k not in config})
        return config
    return DEFAULT_VALUES.copy()


def set_value(key: str,
              value):
    config = read_config()
    config[key] = value
    if not os.path.exists(DIRS.user_config_dir):
        os.makedirs(DIRS.user_config_dir, exist_ok=True)
    with open(CONFIG_FILE, 'w') as configfile:
        yaml.dump(config, configfile)
    _LOGGER.debug(f"Configuration saved to {CONFIG_FILE}.")


def get_value(key: str,
              include_envvars: bool = True):
    if include_envvars:
        if key in ENV_VARS:
            env_var = os.getenv(ENV_VARS[key])
            if env_var is not None:
                return env_var

    config = read_config()
    return config.get(key)


def clear_all_configurations():
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
