import argparse
import logging
from datamint import configs
from datamint.utils.logging_utils import load_cmdline_logging_config
from rich.prompt import Prompt, Confirm
from rich.console import Console

# Create console for user output and logger for developer debugging
console = Console()
_LOGGER = logging.getLogger(__name__)


def configure_default_url():
    """Configure the default API URL interactively."""
    console.print(f"Current default URL: [cyan]{configs.get_value(configs.APIURL_KEY, 'Not set')}[/cyan]")
    url = Prompt.ask("Enter the default API URL (leave empty to abort)").strip()
    if url == '':
        return

    # Basic URL validation
    if not (url.startswith('http://') or url.startswith('https://')):
        console.print("[yellow]⚠️  URL should start with http:// or https://[/yellow]")
        return

    configs.set_value(configs.APIURL_KEY, url)
    console.print("[green]✅ Default API URL set successfully.[/green]")


def ask_api_key(ask_to_save: bool) -> str | None:
    """Ask user for API key with improved guidance."""
    console.print("💡 Get your API key from your Datamint administrator or the web app (https://app.datamint.io/team)")

    api_key = Prompt.ask('API key (leave empty to abort)').strip()
    if api_key == '':
        return None

    if ask_to_save:
        ans = Confirm.ask("Save the API key so it automatically loads next time? (y/n): ",
                          default=True)
        try:
            if ans:
                configs.set_value(configs.APIKEY_KEY, api_key)
                console.print("[green]✅ API key saved.[/green]")
        except Exception as e:
            console.print("[red]❌ Error saving API key.[/red]")
            _LOGGER.exception(e)
    return api_key


def show_all_configurations():
    """Display all current configurations in a user-friendly format."""
    config = configs.read_config()
    if config is not None and len(config) > 0:
        console.print("[bold]📋 Current configurations:[/bold]")
        for key, value in config.items():
            # Mask API key for security
            if key == configs.APIKEY_KEY and value:
                masked_value = f"{value[:3]}...{value[-3:]}" if len(value) > 6 else value
                console.print(f"  [cyan]{key}[/cyan]: [dim]{masked_value}[/dim]")
            else:
                console.print(f"  [cyan]{key}[/cyan]: {value}")
    else:
        console.print("[dim]No configurations found.[/dim]")


def clear_all_configurations():
    """Clear all configurations with confirmation."""
    yesno = Confirm.ask('Are you sure you want to clear all configurations?',
                        default=True)
    if yesno:
        configs.clear_all_configurations()
        console.print("[green]✅ All configurations cleared.[/green]")


def configure_api_key():
    api_key = ask_api_key(ask_to_save=False)
    if api_key is None:
        return
    configs.set_value(configs.APIKEY_KEY, api_key)
    console.print("[green]✅ API key saved.[/green]")


def test_connection():
    """Test the API connection with current settings."""
    try:
        from datamint import APIHandler
        console.print("[blue]🔄 Testing connection...[/blue]")
        api = APIHandler()
        # Simple test - try to get projects
        projects = api.get_projects()
        console.print(f"[green]✅ Connection successful! Found {len(projects)} projects.[/green]")
    except ImportError:
        console.print("[red]❌ Full API not available. Install with: pip install datamint-python-api[full][/red]")
    except Exception as e:
        console.print(f"[red]❌ Connection failed: {e}[/red]")
        console.print("[dim]💡 Check your API key and URL settings[/dim]")


def interactive_mode():
    console.print("[bold blue]🔧 Datamint Configuration Tool[/bold blue]")

    if len(configs.read_config()) == 0:
        console.print("[yellow]👋 Welcome! Let's set up your API key first.[/yellow]")
        configure_api_key()

    while True:
        console.print("\n[bold]📋 Select the action you want to perform:[/bold]")
        console.print(" [cyan](1)[/cyan] Configure the API key")
        console.print(" [cyan](2)[/cyan] Configure the default URL")
        console.print(" [cyan](3)[/cyan] Show all configuration settings")
        console.print(" [cyan](4)[/cyan] Clear all configuration settings")
        console.print(" [cyan](5)[/cyan] Test connection")
        console.print(" [cyan](q)[/cyan] Exit")
        choice = Prompt.ask("Enter your choice").lower().strip()

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
            console.print("[green]👋 Goodbye![/green]")
            break
        else:
            console.print("[red]❌ Invalid choice. Please enter a number between 1 and 5 or 'q' to quit.[/red]")


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
        console.print("[green]✅ API key saved.[/green]")

    if args.default_url is not None:
        # Basic URL validation
        if not (args.default_url.startswith('http://') or args.default_url.startswith('https://')):
            console.print("[red]❌ URL must start with http:// or https://[/red]")
            return
        configs.set_value(configs.APIURL_KEY, args.default_url)
        console.print("[green]✅ Default URL saved.[/green]")

    no_arguments_provided = args.api_key is None and args.default_url is None

    if no_arguments_provided or args.interactive:
        interactive_mode()


if __name__ == "__main__":
    main()
