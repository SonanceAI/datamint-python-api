import argparse
import logging
from datamint import configs
from datamint.utils.logging_utils import load_cmdline_logging_config, ConsoleWrapperHandler
from rich.prompt import Prompt, Confirm
from rich.console import Console
import os
import shutil
from rich.table import Table

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')
console: Console


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
        from datamint import Api
        console.print("[accent]🔄 Testing connection...[/accent]")
        Api(check_connection=True)
        console.print(f"[success]✅ Connection successful![/success]")
    except ImportError:
        console.print("[error]❌ Full API not available. Install with: pip install datamint[/error]")
    except Exception as e:
        console.print(f"[error]❌ Connection failed: {e}[/error]")


def discover_local_datasets() -> list[dict[str, str]]:
    """Discover locally downloaded datasets.
    
    Returns:
        List of dictionaries containing dataset info with keys: 'name', 'path', 'size'
    """
    from datamint.dataset.base_dataset import DatamintBaseDataset
    
    # Check default datamint directory
    default_root = os.path.join(
        os.path.expanduser("~"),
        DatamintBaseDataset.DATAMINT_DEFAULT_DIR,
        DatamintBaseDataset.DATAMINT_DATASETS_DIR
    )
    
    datasets = []
    
    if not os.path.exists(default_root):
        return datasets
    
    for item in os.listdir(default_root):
        dataset_path = os.path.join(default_root, item)
        if os.path.isdir(dataset_path):
            # Check if it has a dataset.json file (indicating it's a datamint dataset)
            dataset_json = os.path.join(dataset_path, 'dataset.json')
            if os.path.exists(dataset_json):
                # Calculate directory size
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(dataset_path)
                    for filename in filenames
                )
                
                datasets.append({
                    'name': item,
                    'path': dataset_path,
                    'size': _format_size(total_size),
                    'size_bytes': total_size
                })
    
    return sorted(datasets, key=lambda x: x['name'])


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def show_local_datasets() -> list[dict[str, str]]:
    """Display all locally downloaded datasets."""
    datasets = discover_local_datasets()
    
    if not datasets:
        console.print("[dim]No local datasets found.[/dim]")
        return datasets
    
    console.print("[title]📁 Local Datasets:[/title]")
    
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Dataset Name", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Path", style="dim")
    
    total_size = 0
    for dataset in datasets:
        table.add_row(dataset['name'], dataset['size'], dataset['path'])
        total_size += dataset['size_bytes']
    
    console.print(table)
    console.print(f"\n[bold]Total size:[/bold] {_format_size(total_size)}")
    
    return datasets


def clean_dataset(dataset_name: str) -> bool:
    """Clean a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to clean
        
    Returns:
        True if dataset was cleaned, False otherwise
    """
    datasets = discover_local_datasets()
    dataset_to_clean = None
    
    for dataset in datasets:
        if dataset['name'] == dataset_name:
            dataset_to_clean = dataset
            break
    
    if dataset_to_clean is None:
        console.print(f"[error]❌ Dataset '{dataset_name}' not found locally.[/error]")
        return False
    
    console.print(f"[warning]⚠️  About to delete dataset: {dataset_name}[/warning]")
    console.print(f"[dim]Path: {dataset_to_clean['path']}[/dim]")
    console.print(f"[dim]Size: {dataset_to_clean['size']}[/dim]")
    
    confirmed = Confirm.ask("Are you sure you want to delete this dataset?", 
                           default=False, console=console)
    
    if not confirmed:
        console.print("[dim]Operation cancelled.[/dim]")
        return False
    
    try:
        shutil.rmtree(dataset_to_clean['path'])
        console.print(f"[success]✅ Dataset '{dataset_name}' has been deleted.[/success]")
        return True
    except Exception as e:
        console.print(f"[error]❌ Error deleting dataset: {e}[/error]")
        _LOGGER.exception(e)
        return False


def clean_all_datasets() -> bool:
    """Clean all locally downloaded datasets.
    
    Returns:
        True if datasets were cleaned, False otherwise
    """
    datasets = discover_local_datasets()
    
    if not datasets:
        console.print("[dim]No local datasets found to clean.[/dim]")
        return True
    
    console.print(f"[warning]⚠️  About to delete {len(datasets)} dataset(s):[/warning]")
    
    table = Table(show_header=True, header_style="bold red")
    table.add_column("Dataset Name", style="cyan")
    table.add_column("Size", justify="right", style="green")
    
    total_size = 0
    for dataset in datasets:
        table.add_row(dataset['name'], dataset['size'])
        total_size += dataset['size_bytes']
    
    console.print(table)
    console.print(f"\n[bold red]Total size to be deleted:[/bold red] {_format_size(total_size)}")

    confirmed = Confirm.ask("Are you sure you want to delete ALL local datasets? (this does not affect remote datasets)", 
                           default=False, console=console)
    
    if not confirmed:
        console.print("[dim]Operation cancelled.[/dim]")
        return False
    
    success_count = 0
    for dataset in datasets:
        try:
            shutil.rmtree(dataset['path'])
            console.print(f"[success]✅ Deleted: {dataset['name']}[/success]")
            success_count += 1
        except Exception as e:
            console.print(f"[error]❌ Failed to delete {dataset['name']}: {e}[/error]")
            _LOGGER.exception(e)
    
    if success_count == len(datasets):
        console.print(f"[success]✅ Successfully deleted all {success_count} datasets.[/success]")
        return True
    else:
        console.print(f"[warning]⚠️  Deleted {success_count} out of {len(datasets)} datasets.[/warning]")
        return False


def interactive_dataset_cleaning() -> None:
    """Interactive dataset cleaning menu."""
    datasets = show_local_datasets()
    
    if not datasets:
        return
    
    console.print("\n[title]🧹 Dataset Cleaning Options:[/title]")
    console.print(" [accent](1)[/accent] Clean a specific dataset")
    console.print(" [accent](2)[/accent] Clean all datasets")
    console.print(" [accent](b)[/accent] Back to main menu")

    try:
        choice = Prompt.ask("Enter your choice", console=console).lower().strip()
        
        # Handle ESC key (appears as escape sequence)
        if choice in ('', '\x1b', 'esc', 'escape'):
            return
        
        if choice == '1':
            dataset_names = [d['name'] for d in datasets]
            console.print("\n[title]Available datasets:[/title]")
            for i, name in enumerate(dataset_names, 1):
                console.print(f" [accent]({i})[/accent] {name}")
            
            dataset_choice = Prompt.ask("Enter dataset number or name", console=console).strip()
            
            # Handle ESC key in dataset selection
            if dataset_choice in ('', '\x1b', 'esc', 'escape'):
                return
            
            # Handle numeric choice
            try:
                dataset_idx = int(dataset_choice) - 1
                if 0 <= dataset_idx < len(dataset_names):
                    clean_dataset(dataset_names[dataset_idx])
                    return
            except ValueError:
                pass
            
            # Handle name choice
            if dataset_choice in dataset_names:
                clean_dataset(dataset_choice)
            else:
                console.print("[error]❌ Invalid dataset selection.[/error]")
                
        elif choice == '2':
            clean_all_datasets()
        elif choice != 'b':
            console.print("[error]❌ Invalid choice.[/error]")
    except KeyboardInterrupt:
        pass


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
            console.print(" [accent](6)[/accent] Manage/Show local datasets...")
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
            elif choice == '6':
                interactive_dataset_cleaning()
            elif choice in ('q', 'exit', 'quit'):
                break
            else:
                console.print("[error]❌ Invalid choice. Please enter a number between 1 and 7 or 'q' to quit.[/error]")
    except KeyboardInterrupt:
        console.print('')

    console.print("[success]👋 Goodbye![/success]")


def main():
    """Main entry point for the configuration tool."""
    global console
    load_cmdline_logging_config()
    console = [h for h in _USER_LOGGER.handlers if isinstance(h, ConsoleWrapperHandler)][0].console
    parser = argparse.ArgumentParser(
        description='🔧 Datamint API Configuration Tool',
        epilog="""
Examples:
  datamint-config                           # Interactive mode
  datamint-config --api-key YOUR_KEY        # Set API key
  datamint-config --list-datasets           # Show local datasets
  datamint-config --clean-dataset NAME      # Clean specific dataset
  datamint-config --clean-all-datasets      # Clean all datasets
  
More Documentation: https://sonanceai.github.io/datamint-python-api/command_line_tools.html
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--api-key', type=str, help='API key to set')
    parser.add_argument('--default-url', '--url', type=str, help='Default URL to set')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode (default if no other arguments provided)')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List all locally downloaded datasets')
    parser.add_argument('--clean-dataset', type=str, metavar='DATASET_NAME',
                        help='Clean a specific dataset by name')
    parser.add_argument('--clean-all-datasets', action='store_true',
                        help='Clean all locally downloaded datasets')

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

    if args.list_datasets:
        show_local_datasets()

    if args.clean_dataset:
        clean_dataset(args.clean_dataset)

    if args.clean_all_datasets:
        clean_all_datasets()

    no_arguments_provided = (args.api_key is None and args.default_url is None and
                           not args.list_datasets and not args.clean_dataset and
                           not args.clean_all_datasets)

    if no_arguments_provided or args.interactive:
        interactive_mode()


if __name__ == "__main__":
    main()
