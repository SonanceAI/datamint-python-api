import argparse
import logging
from datamintapi import configs
from datamintapi.utils.logging_utils import load_cmdline_logging_config

# Create two loggings: one for the user and one for the developer
_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')


def configure_default_url():
    """Configure the default API URL interactively."""
    _USER_LOGGER.info("Current default URL: %s", configs.get_value(configs.APIURL_KEY, 'Not set'))
    url = input("Enter the default API URL (leave empty to abort): ").strip()
    if url == '':
        return

    # Basic URL validation
    if not (url.startswith('http://') or url.startswith('https://')):
        _USER_LOGGER.warning("URL should start with http:// or https://")
        return

    configs.set_value(configs.APIURL_KEY, url)
    _USER_LOGGER.info("Default API URL set successfully.")


def ask_api_key(ask_to_save: bool) -> str | None:
    """Ask user for API key with improved guidance."""
    _USER_LOGGER.info("💡 Get your API key from your Datamint administrator or the web app (https://app.datamint.io/team)")

    api_key = input('API key (leave empty to abort): ').strip()
    if api_key == '':
        return None

    if ask_to_save:
        ans = input("Save the API key so it automatically loads next time? (y/n): ")
        try:
            if ans.lower() == 'y':
                configs.set_value(configs.APIKEY_KEY, api_key)
                _USER_LOGGER.info("✅ API key saved.")
        except Exception as e:
            _USER_LOGGER.error("❌ Error saving API key.")
            _LOGGER.exception(e)
    return api_key


def show_all_configurations():
    """Display all current configurations in a user-friendly format."""
    config = configs.read_config()
    if config is not None and len(config) > 0:
        _USER_LOGGER.info("📋 Current configurations:")
        for key, value in config.items():
            # Mask API key for security
            if key == configs.APIKEY_KEY and value:
                masked_value = f"{value[:3]}...{value[-3:]}" if len(value) > 6 else value
                _USER_LOGGER.info(f"  {key}: {masked_value}")
            else:
                _USER_LOGGER.info(f"  {key}: {value}")
    else:
        _USER_LOGGER.info("No configurations found.")


def clear_all_configurations():
    """Clear all configurations with confirmation."""
    yesno = input('Are you sure you want to clear all configurations? (y/n): ')
    if yesno.lower() == 'y':
        configs.clear_all_configurations()
        _USER_LOGGER.info("All configurations cleared.")


def configure_api_key():
    api_key = ask_api_key(ask_to_save=False)
    if api_key is None:
        return
    configs.set_value(configs.APIKEY_KEY, api_key)
    _USER_LOGGER.info("✅ API key saved.")


def test_connection():
    """Test the API connection with current settings."""
    try:
        from datamintapi import APIHandler
        _USER_LOGGER.info("🔄 Testing connection...")
        api = APIHandler()
        # Simple test - try to get projects
        projects = api.get_projects()
        _USER_LOGGER.info(f"✅ Connection successful! Found {len(projects)} projects.")
    except ImportError:
        _USER_LOGGER.error("❌ Full API not available. Install with: pip install datamint-python-api[full]")
    except Exception as e:
        _USER_LOGGER.error(f"❌ Connection failed: {e}")
        _USER_LOGGER.info("💡 Check your API key and URL settings")


def interactive_mode():
    _USER_LOGGER.info("🔧 Datamint Configuration Tool")

    if len(configs.read_config()) == 0:
        _USER_LOGGER.info("👋 Welcome! Let's set up your API key first.")
        configure_api_key()

    while True:
        _USER_LOGGER.info("\n📋 Select the action you want to perform:")
        _USER_LOGGER.info(" (1) Configure the API key")
        _USER_LOGGER.info(" (2) Configure the default URL")
        _USER_LOGGER.info(" (3) Show all configuration settings")
        _USER_LOGGER.info(" (4) Clear all configuration settings")
        _USER_LOGGER.info(" (5) Test connection")
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
        elif choice == '5':
            test_connection()
        elif choice in ('q', 'exit', 'quit'):
            _USER_LOGGER.info("👋 Goodbye!")
            break
        else:
            _USER_LOGGER.info("❌ Invalid choice. Please enter a number between 1 and 5 or 'q' to quit.")


def main():
    load_cmdline_logging_config()
    parser = argparse.ArgumentParser(
        description='🔧 Datamint API Configuration Tool',
        epilog="""
Examples:
  datamint-config                           # Interactive mode
  datamint-config --api-key YOUR_KEY        # Set API key
  
More Documentation: https://sonanceai.github.io/datamint-python-api/command_line_tools.html
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--api-key', type=str, help='API key to set')
    parser.add_argument('--default-url', '--url', type=str, help='Default URL to set')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode (default if no other arguments provided)')

    args = parser.parse_args()

    if args.api_key is not None:
        configs.set_value(configs.APIKEY_KEY, args.api_key)
        _USER_LOGGER.info("✅ API key saved.")

    if args.default_url is not None:
        # Basic URL validation
        if not (args.default_url.startswith('http://') or args.default_url.startswith('https://')):
            _USER_LOGGER.error("❌ URL must start with http:// or https://")
            return
        configs.set_value(configs.APIURL_KEY, args.default_url)
        _USER_LOGGER.info("✅ Default URL saved.")

    no_arguments_provided = args.api_key is None and args.default_url is None

    if no_arguments_provided or args.interactive:
        interactive_mode()


if __name__ == "__main__":
    main()
