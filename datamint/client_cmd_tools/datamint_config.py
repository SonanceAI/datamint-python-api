import argparse
import logging
import os
import platform
from datamint import configs
from datamint.utils.logging_utils import load_cmdline_logging_config
from rich.prompt import Prompt, Confirm
from rich.console import Console
from rich.theme import Theme

# Create a custom theme that works well on both dark and blue backgrounds
def _create_console_theme() -> Theme:
    """Create a custom Rich theme optimized for cross-platform terminals."""
    # Detect if we're likely on PowerShell (Windows + PowerShell)
    is_powershell = (
        platform.system() == "Windows" and 
        os.environ.get("PSModulePath") is not None
    )
    
    if is_powershell:
        # PowerShell blue background - use high contrast colors
        return Theme({
            "warning": "bright_yellow",
            "error": "bright_red on white",
            "success": "bright_green",
            "key": "bright_cyan",
            "accent": "bright_cyan",
            "title": "bold"
        })
    else:
        # Linux/Unix terminals - standard colors
        return Theme({
            "warning": "yellow",
            "error": "red",
            "success": "green",
            "key": "cyan",
            "accent": "bright_blue",
            "title": "bold"
        })

# Create console with custom theme
console = Console(theme=_create_console_theme())
_LOGGER = logging.getLogger(__name__)


def configure_default_url():
    """Configure the default API URL interactively."""
    current_url = configs.get_value(configs.APIURL_KEY, 'Not set')
    console.print(f"Current default URL: [key]{current_url}[/key]")
    url = Prompt.ask("Enter the default API URL (leave empty to abort)", console=console).strip()
    if url == '':
        return

    # Basic URL validation
    if not (url.startswith('http://') or url.startswith('https://')):
        console.print("[warning]⚠️  URL should start with http:// or https://[/warning]")
        return

    configs.set_value(configs.APIURL_KEY, url)
    console.print("[success]✅ Default API URL set successfully.[/success]")


def ask_api_key(ask_to_save: bool) -> str | None:
    """Ask user for API key with improved guidance."""
    console.print("[info]💡 Get your API key from your Datamint administrator or the web app (https://app.datamint.io/team)[/info]")

    api_key = Prompt.ask('API key (leave empty to abort)', console=console).strip()
    if api_key == '':
        return None

    if ask_to_save:
        ans = Confirm.ask("Save the API key so it automatically loads next time? (y/n): ",
                          default=True, console=console)
        try:
            if ans:
                configs.set_value(configs.APIKEY_KEY, api_key)
                console.print("[success]✅ API key saved.[/success]")
        except Exception as e:
            console.print("[error]❌ Error saving API key.[/error]")
            _LOGGER.exception(e)
    return api_key


def show_all_configurations():
    """Display all current configurations in a user-friendly format."""
    config = configs.read_config()
    if config is not None and len(config) > 0:
        console.print("[title]📋 Current configurations:[/title]")
        for key, value in config.items():
            # Mask API key for security
            if key == configs.APIKEY_KEY and value:
                masked_value = f"{value[:3]}...{value[-3:]}" if len(value) > 6 else value
                console.print(f"  [key]{key}[/key]: [dim]{masked_value}[/dim]")
            else:
                console.print(f"  [key]{key}[/key]: {value}")
    else:
        console.print("[dim]No configurations found.[/dim]")


def clear_all_configurations():
    """Clear all configurations with confirmation."""
    yesno = Confirm.ask('Are you sure you want to clear all configurations?',
                        default=True, console=console)
    if yesno:
        configs.clear_all_configurations()
        console.print("[success]✅ All configurations cleared.[/success]")


def configure_api_key():
    """Configure API key interactively."""
    api_key = ask_api_key(ask_to_save=False)
    if api_key is None:
        return
    configs.set_value(configs.APIKEY_KEY, api_key)
    console.print("[success]✅ API key saved.[/success]")


def test_connection():
    """Test the API connection with current settings."""
    try:
        from datamint import APIHandler
        console.print("[accent]🔄 Testing connection...[/accent]")
        api = APIHandler()
        # Simple test - try to get projects
        projects = api.get_projects()
        console.print(f"[success]✅ Connection successful! Found {len(projects)} projects.[/success]")
    except ImportError:
        console.print("[error]❌ Full API not available. Install with: pip install datamint-python-api[full][/error]")
    except Exception as e:
        console.print(f"[error]❌ Connection failed: {e}[/error]")


def interactive_mode():
    """Run the interactive configuration mode."""
    console.print("[title]🔧 Datamint Configuration Tool[/title]")

    try:
        if len(configs.read_config()) == 0:
            console.print("[warning]👋 Welcome! Let's set up your API key first.[/warning]")
            configure_api_key()

        while True:
            console.print("\n[title]📋 Select the action you want to perform:[/title]")
            console.print(" [accent](1)[/accent] Configure the API key")
            console.print(" [accent](2)[/accent] Configure the default URL")
            console.print(" [accent](3)[/accent] Show all configuration settings")
            console.print(" [accent](4)[/accent] Clear all configuration settings")
            console.print(" [accent](5)[/accent] Test connection")
            console.print(" [accent](q)[/accent] Exit")
            choice = Prompt.ask("Enter your choice", console=console).lower().strip()

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
                break
            else:
                console.print("[error]❌ Invalid choice. Please enter a number between 1 and 5 or 'q' to quit.[/error]")
    except KeyboardInterrupt:
        console.print('')

    console.print("[success]👋 Goodbye![/success]")

def main():
    """Main entry point for the configuration tool."""
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
        console.print("[success]✅ API key saved.[/success]")

    if args.default_url is not None:
        # Basic URL validation
        if not (args.default_url.startswith('http://') or args.default_url.startswith('https://')):
            console.print("[error]❌ URL must start with http:// or https://[/error]")
            return
        configs.set_value(configs.APIURL_KEY, args.default_url)
        console.print("[success]✅ Default URL saved.[/success]")

    no_arguments_provided = args.api_key is None and args.default_url is None

    if no_arguments_provided or args.interactive:
        interactive_mode()


if __name__ == "__main__":
    main()
