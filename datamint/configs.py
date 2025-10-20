import yaml
import os
import logging
from netrc import netrc
from platformdirs import PlatformDirs
from typing import Dict
from pathlib import Path

APIURL_KEY = 'default_api_url'
APIKEY_KEY = 'api_key'

ENV_VARS = {
    APIKEY_KEY: 'DATAMINT_API_KEY',
    APIURL_KEY: 'DATAMINT_API_URL'
}

_LOGGER = logging.getLogger(__name__)

DIRS = PlatformDirs(appname='datamintapi')
CONFIG_FILE = os.path.join(DIRS.user_config_dir, 'datamintapi.yaml')


def get_env_var_name(key: str) -> str:
    return ENV_VARS[key]

def read_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as configfile:
            return yaml.safe_load(configfile)
    return {}


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
