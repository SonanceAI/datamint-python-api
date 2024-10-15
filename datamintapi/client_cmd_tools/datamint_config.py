import argparse
import logging
from datamintapi import configs
from datamintapi.utils.logging_utils import load_cmdline_logging_config

# Create two loggings: one for the user and one for the developer
_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')


def configure_default_url():
    url = input("Enter the default API URL (leave empty to abort): ").strip()
    if url == '':
        return
    configs.set_value(configs.APIURL_KEY, url)
    _USER_LOGGER.info("Default API URL set sucessufully.")


def ask_api_key(ask_to_save: bool) -> str:
    api_key = input('API key (leave empty to abort): ').strip()
    if api_key == '':
        return None

    if ask_to_save:
        ans = input("Save the API key so it automatically loads next time? (y/n): ")
        try:
            if ans.lower() == 'y':
                configs.set_value(configs.APIKEY_KEY, api_key)
                _USER_LOGGER.info(f"API key saved.")
        except Exception as e:
            _USER_LOGGER.error(f"Error saving API key.")
            _LOGGER.exception(e)
    return api_key


def show_all_configurations():
    config = configs.read_config()
    if config is not None and len(config) > 0:
        _USER_LOGGER.info("Current configurations:")
        for key, value in config.items():
            _USER_LOGGER.info(f"  {key}: {value}")
    else:
        _USER_LOGGER.info("No configurations found.")


def clear_all_configurations():
    yesno = input('Are you sure you want to clear all configurations? (y/n): ')
    if yesno.lower() == 'y':
        configs.clear_all_configurations()
        _USER_LOGGER.info("All configurations cleared.")


def configure_api_key():
    api_key = ask_api_key(ask_to_save=False)
    if api_key is None:
        return
    configs.set_value(configs.APIKEY_KEY, api_key)
    _USER_LOGGER.info(f"API key saved.")


def interactive_mode():
    if len(configs.read_config()) == 0:
        configure_api_key()
    while True:
        _USER_LOGGER.info("\nSelect the action you want to perform:")
        _USER_LOGGER.info(" (1) Configure the API key")
        _USER_LOGGER.info(" (2) Configure the default URL")
        _USER_LOGGER.info(" (3) Show all configuration settings")
        _USER_LOGGER.info(" (4) Clear all configuration settings")
        _USER_LOGGER.info(" (q) Exit")
        choice = input("Enter your choice: ").lower().strip()
        if choice == '1':
            configure_api_key()
        elif choice == '2':
            configure_default_url()
        elif choice == '3':
            show_all_configurations()
        elif choice == '4':
            clear_all_configurations()
        elif choice in ('5', 'q', 'exit', 'quit'):
            break
        else:
            _USER_LOGGER.info("Invalid choice. Please enter a number between 1 and 4 or 'q' to quit.")


def main():
    load_cmdline_logging_config()
    parser = argparse.ArgumentParser(description='DatamintAPI command line tool for configurations')
    parser.add_argument('--api-key', type=str, help='API key to set')
    parser.add_argument('--default-url', '--url', type=str, help='Default URL to set')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode. Providing no other arguments will also start the interactive mode.')

    args = parser.parse_args()

    if args.api_key is not None:
        configs.set_value(configs.APIKEY_KEY, args.api_key)
        _USER_LOGGER.info("API key saved.")
    if args.default_url is not None:
        configs.set_value(configs.APIURL_KEY, args.default_url)
        _USER_LOGGER.info("Default URL saved.")

    no_arguments_provided = args.api_key is None and args.default_url is None

    if no_arguments_provided or args.interactive == True:
        interactive_mode()


if __name__ == "__main__":
    main()
