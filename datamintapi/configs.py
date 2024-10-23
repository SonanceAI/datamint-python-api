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

_NETRC_MACHINE_KEY = 'api.datamint.io'

_LOGGER = logging.getLogger(__name__)

DIRS = PlatformDirs(appname='datamintapi')
CONFIG_FILE = os.path.join(DIRS.user_config_dir, 'datamintapi.yaml')


def read_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as configfile:
            return yaml.safe_load(configfile)
    return {}

### NETRC. Not used ###
# def _set_api_key(api_key: str):
#     api_key = _get_netrc_api_key()
#     if api_key is not None:
#         _clear_api_key()
#     netrc_file = Path.home() / ".netrc"
#     with open(netrc_file, 'a') as f:
#         f.write(f"\nmachine {_NETRC_MACHINE_KEY}  login user  password {api_key}\n")
#     _LOGGER.debug(f"API key saved to {netrc_file}.")

# def _clear_api_key():
#     netrc_file = Path.home() / ".netrc"
#     if netrc_file.exists():
#         with open(netrc_file, 'r') as f:
#             lines = f.readlines()
#         with open(netrc_file, 'w') as f:
#             for line in lines:
#                 if _NETRC_MACHINE_KEY not in line:
#                     f.write(line)


def _get_netrc_api_key() -> str:
    netrc_file = Path.home() / ".netrc"
    if netrc_file.exists():
        token = netrc(netrc_file).authenticators(_NETRC_MACHINE_KEY)
        if token is not None:
            return token[2]
    return None


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

    if key == APIKEY_KEY:
        try:
            api_key = _get_netrc_api_key()
            if api_key is not None:
                _LOGGER.info("API key loaded from netrc file.")
                return api_key
        except Exception as e:
            _LOGGER.info(f"Error reading API key from .netrc file: {e}.")

    config = read_config()
    return config.get(key)


def clear_all_configurations():
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
