import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import NotRequired, TypedDict

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from datamint import configs
from datamint.utils.logging_utils import ConsoleWrapperHandler, load_cmdline_logging_config

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')
console: Console = Console()

_CACHE_NAMESPACE_KIND = 'cache namespace'
_LEGACY_DATASET_KIND = 'legacy dataset'
_UPLOAD_CHANNEL_KIND = 'upload channel'
_TAG_KIND = 'tag'
_LEGACY_DATASET_ROOT_NAMES = ('datasets', 'datasets_old')


class LocalDataGroup(TypedDict):
    identifier: str
    name: str
    kind: str
    path: str
    size: str
    size_bytes: int
    item_count: int
    resource_dirs: NotRequired[list[str]]


class ResourceFilterAccumulator(TypedDict):
    size_bytes: int
    resource_dirs: list[str]


def configure_default_url():
    """Configure the default API URL interactively."""
    current_url = configs.get_value(configs.APIURL_KEY) or 'Not set'
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


def _get_local_data_root() -> Path | None:
    """Return the local Datamint data root if configured."""
    if not configs.DATAMINT_DATA_DIR:
        return None
    return Path(configs.DATAMINT_DATA_DIR).expanduser()


def _get_resources_cache_root() -> Path | None:
    """Return the resources cache root if it exists."""
    data_root = _get_local_data_root()
    if data_root is None:
        return None

    resources_root = data_root / 'resources'
    if not resources_root.exists() or not resources_root.is_dir():
        return None

    return resources_root


def _calculate_directory_size(path: Path) -> int:
    """Calculate the recursive size of a directory in bytes."""
    total_size = 0
    try:
        for child in path.rglob('*'):
            if child.is_file():
                total_size += child.stat().st_size
    except OSError as exc:
        _LOGGER.warning("Failed to inspect local data size for %s: %s", path, exc)
    return total_size


def _count_direct_children(path: Path) -> int:
    """Count direct children of a directory."""
    try:
        return sum(1 for _ in path.iterdir())
    except OSError as exc:
        _LOGGER.warning("Failed to inspect local data entries for %s: %s", path, exc)
        return 0


def _get_legacy_dataset_entry_count(dataset_path: Path) -> int:
    """Return the number of resources recorded in a legacy dataset folder."""
    dataset_json = dataset_path / 'dataset.json'
    if not dataset_json.exists():
        return _count_direct_children(dataset_path)

    try:
        dataset_data = json.loads(dataset_json.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        _LOGGER.warning("Failed to read legacy dataset metadata for %s: %s", dataset_path, exc)
        return _count_direct_children(dataset_path)

    resource_ids = dataset_data.get('resource_ids')
    if isinstance(resource_ids, list):
        return len(resource_ids)

    resources = dataset_data.get('resources')
    if isinstance(resources, list):
        return len(resources)

    return _count_direct_children(dataset_path)


def _make_local_data_group_identifier(kind: str, name: str) -> str:
    """Create a stable selector for a local data group."""
    if kind == _CACHE_NAMESPACE_KIND:
        prefix = 'cache'
    elif kind == _LEGACY_DATASET_KIND:
        prefix = 'legacy'
    elif kind == _UPLOAD_CHANNEL_KIND:
        prefix = 'channel'
    elif kind == _TAG_KIND:
        prefix = 'tag'
    else:
        prefix = 'project-resources'
    return f"{prefix}:{name}"


def _build_local_data_group(
    name: str,
    kind: str,
    path: Path,
    item_count: int,
) -> LocalDataGroup:
    """Build a serializable local data group summary."""
    size_bytes = _calculate_directory_size(path)
    return {
        'identifier': _make_local_data_group_identifier(kind, name),
        'name': name,
        'kind': kind,
        'path': str(path),
        'size': _format_size(size_bytes),
        'size_bytes': size_bytes,
        'item_count': item_count,
    }


def _build_resource_filter_group(
    name: str,
    kind: str,
    resources_root: Path,
    resource_dirs: list[str],
    size_bytes: int,
) -> LocalDataGroup:
    """Build a virtual local data group backed by a subset of cached resources."""
    return {
        'identifier': _make_local_data_group_identifier(kind, name),
        'name': name,
        'kind': kind,
        'path': str(resources_root),
        'size': _format_size(size_bytes),
        'size_bytes': size_bytes,
        'item_count': len(resource_dirs),
        'resource_dirs': resource_dirs,
    }


def discover_local_datasets() -> list[LocalDataGroup]:
    """Discover local Datamint cache namespaces and legacy dataset folders.

    Active local data is stored in top-level namespaces such as ``resources``
    and ``annotations``. Legacy dataset downloads are still reported from
    ``datasets`` or ``datasets_old``. Resource subsets grouped by upload
    channel or tag are discovered separately because they overlap with the
    top-level ``resources`` namespace.

    Returns:
        List of dictionaries describing local data groups.
    """
    data_root = _get_local_data_root()
    if data_root is None or not data_root.exists():
        return []

    groups: list[LocalDataGroup] = []

    for item in sorted(data_root.iterdir(), key=lambda path: path.name.lower()):
        if not item.is_dir():
            continue

        if item.name in _LEGACY_DATASET_ROOT_NAMES:
            for dataset_dir in sorted(item.iterdir(), key=lambda path: path.name.lower()):
                if not dataset_dir.is_dir():
                    continue

                group = _build_local_data_group(
                    name=dataset_dir.name,
                    kind=_LEGACY_DATASET_KIND,
                    path=dataset_dir,
                    item_count=_get_legacy_dataset_entry_count(dataset_dir),
                )
                if group['item_count'] > 0 or group['size_bytes'] > 0:
                    groups.append(group)
            continue

        group = _build_local_data_group(
            name=item.name,
            kind=_CACHE_NAMESPACE_KIND,
            path=item,
            item_count=_count_direct_children(item),
        )
        if group['item_count'] > 0 or group['size_bytes'] > 0:
            groups.append(group)

    return groups


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size_value = float(size_bytes)
    while size_value >= 1024 and i < len(size_names) - 1:
        size_value /= 1024.0
        i += 1
    
    return f"{size_value:.1f} {size_names[i]}"


def _discover_resource_filter_groups(
    *,
    tag_limit: int | None = None,
) -> tuple[list[LocalDataGroup], list[LocalDataGroup]]:
    """Discover cleanable cached-resource groups derived from channel and tag metadata."""
    from datamint.entities.cache_manager import CacheManager

    resources_root = _get_resources_cache_root()
    if resources_root is None:
        return [], []

    cache_mgr = CacheManager('resources')
    channel_matches: dict[str, ResourceFilterAccumulator] = {}
    tag_matches: dict[str, ResourceFilterAccumulator] = {}

    def add_match(
        groups: dict[str, ResourceFilterAccumulator],
        group_name: str,
        resource_dir: Path,
        resource_size: int,
    ) -> None:
        group = groups.setdefault(group_name, {'size_bytes': 0, 'resource_dirs': []})
        group['size_bytes'] += resource_size
        group['resource_dirs'].append(str(resource_dir))

    def finalize_groups(
        groups: dict[str, ResourceFilterAccumulator],
        kind: str,
    ) -> list[LocalDataGroup]:
        discovered_groups: list[LocalDataGroup] = []

        for name, group_data in groups.items():
            resource_dirs = group_data['resource_dirs']
            if not resource_dirs:
                continue

            discovered_groups.append(
                _build_resource_filter_group(
                    name=name,
                    kind=kind,
                    resources_root=resources_root,
                    resource_dirs=resource_dirs,
                    size_bytes=group_data['size_bytes'],
                )
            )

        discovered_groups.sort(
            key=lambda group: (-group['item_count'], group['name'].lower())
        )
        return discovered_groups

    for entity_id, extra_info in cache_mgr.iter_entities_extra_info():
        resource_dir = resources_root / entity_id
        if not resource_dir.exists() or not resource_dir.is_dir():
            continue

        resource_size = _calculate_directory_size(resource_dir)

        channel_name = extra_info.get('upload_channel')
        if isinstance(channel_name, str):
            channel_name = channel_name.strip()
            if channel_name:
                add_match(channel_matches, channel_name, resource_dir, resource_size)

        tags = extra_info.get('tags')
        if isinstance(tags, list):
            normalized_tags = {
                str(tag).strip() for tag in tags if isinstance(tag, str) and str(tag).strip()
            }
            for tag_name in normalized_tags:
                add_match(tag_matches, tag_name, resource_dir, resource_size)

    channel_groups = finalize_groups(channel_matches, _UPLOAD_CHANNEL_KIND)
    tag_groups = finalize_groups(tag_matches, _TAG_KIND)
    if tag_limit is not None:
        tag_groups = tag_groups[:tag_limit]

    return channel_groups, tag_groups


def _discover_cleanable_local_data_groups(
    *,
    tag_limit: int | None = None,
) -> list[LocalDataGroup]:
    """Return all top-level groups plus cleanable channel/tag resource subsets."""
    base_groups = discover_local_datasets()
    channel_groups, tag_groups = _discover_resource_filter_groups(tag_limit=tag_limit)
    return [*base_groups, *channel_groups, *tag_groups]


def _render_resource_filter_groups_table(
    groups: list[LocalDataGroup],
    *,
    name_header: str,
    header_style: str,
) -> None:
    """Render a table for channel/tag-backed resource filter groups."""
    table = Table(show_header=True, header_style=header_style)
    table.add_column(name_header, style="cyan")
    table.add_column("Resources", justify="right", style="yellow")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Selector", style="dim")

    for group in groups:
        table.add_row(
            str(group['name']),
            str(group['item_count']),
            str(group['size']),
            str(group['identifier']),
        )

    console.print(table)


def _render_local_data_groups_table(
    groups: list[LocalDataGroup],
    *,
    header_style: str,
    include_kind: bool = False,
) -> None:
    """Render a table of local data groups."""
    table = Table(show_header=True, header_style="bold blue")
    table.header_style = header_style
    table.add_column("Name", style="cyan")
    if include_kind:
        table.add_column("Kind", style="magenta")
    table.add_column("Entries", justify="right", style="yellow")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Path", style="dim")

    for group in groups:
        row = [
            str(group['name']),
            str(group['item_count']),
            str(group['size']),
            str(group['path']),
        ]
        if include_kind:
            row.insert(1, str(group['kind']))
        table.add_row(*row)

    console.print(table)


def show_local_datasets() -> list[LocalDataGroup]:
    """Display top-level local data plus cleanable resource filter summaries."""
    datasets = discover_local_datasets()

    if not datasets:
        console.print("[dim]No local data groups found.[/dim]")
        return datasets

    console.print("[title]📁 Local Data Groups:[/title]")

    cache_groups = [group for group in datasets if group['kind'] == _CACHE_NAMESPACE_KIND]
    legacy_groups = [group for group in datasets if group['kind'] == _LEGACY_DATASET_KIND]

    if cache_groups:
        console.print("\n[title]📦 Cache Namespaces:[/title]")
        _render_local_data_groups_table(cache_groups, header_style="bold blue")

    if legacy_groups:
        console.print("\n[title]🗂️  Legacy Dataset Folders:[/title]")
        _render_local_data_groups_table(legacy_groups, header_style="bold blue")

    channel_groups, tag_groups = _discover_resource_filter_groups(tag_limit=10)
    if channel_groups:
        console.print("\n[title]📡 Resources by Upload Channel:[/title]")
        _render_resource_filter_groups_table(
            channel_groups,
            name_header="Upload Channel",
            header_style="bold blue",
        )

    if tag_groups:
        console.print("\n[title]🏷️  Resources by Tag (Top 10):[/title]")
        _render_resource_filter_groups_table(
            tag_groups,
            name_header="Tag",
            header_style="bold blue",
        )

    total_size = sum(group['size_bytes'] for group in datasets)
    console.print(f"\n[bold]Total size:[/bold] {_format_size(total_size)}")

    return datasets


def _find_local_data_group(
    group_name: str,
    groups: list[LocalDataGroup],
) -> LocalDataGroup | None:
    """Resolve a local data group by selector or display name."""
    normalized_name = group_name.strip().lower()
    identifier_matches = [
        group for group in groups if str(group['identifier']).lower() == normalized_name
    ]
    if len(identifier_matches) == 1:
        return identifier_matches[0]

    name_matches = [group for group in groups if str(group['name']).lower() == normalized_name]
    if len(name_matches) == 1:
        return name_matches[0]

    if len(name_matches) > 1:
        console.print("[error]❌ Group name is ambiguous. Use one of these selectors:[/error]")
        for group in name_matches:
            console.print(f" [dim]- {group['identifier']}[/dim]")

    return None


def clean_dataset(dataset_name: str) -> bool:
    """Clean a specific local data group.

    Args:
        dataset_name: Name or selector of the local data group to clean

    Returns:
        True if the group was cleaned, False otherwise
    """
    datasets = _discover_cleanable_local_data_groups()
    dataset_to_clean = _find_local_data_group(dataset_name, datasets)

    if dataset_to_clean is None:
        console.print(f"[error]❌ Local data group '{dataset_name}' not found.[/error]")
        return False

    console.print(
        f"[warning]⚠️  About to delete {dataset_to_clean['kind']}: {dataset_to_clean['name']}[/warning]"
    )
    console.print(f"[dim]Selector: {dataset_to_clean['identifier']}[/dim]")
    console.print(f"[dim]Path: {dataset_to_clean['path']}[/dim]")
    console.print(f"[dim]Size: {dataset_to_clean['size']}[/dim]")
    console.print("[dim]This only removes local data and does not affect remote resources or annotations.[/dim]")

    confirmed = Confirm.ask(
        "Are you sure you want to delete this local data group?",
        default=False,
        console=console,
    )

    if not confirmed:
        console.print("[dim]Operation cancelled.[/dim]")
        return False

    try:
        resource_dirs = dataset_to_clean.get('resource_dirs')
        if isinstance(resource_dirs, list):
            deleted_count = 0
            for rd in resource_dirs:
                try:
                    shutil.rmtree(rd)
                    deleted_count += 1
                except Exception as exc:
                    console.print(f"[error]❌ Failed to delete {rd}: {exc}[/error]")
                    _LOGGER.warning("Failed to delete %s: %s", rd, exc)
            console.print(
                f"[success]✅ Deleted {deleted_count} cached resource(s) for "
                f"'{dataset_to_clean['name']}'.[/success]"
            )
        else:
            shutil.rmtree(str(dataset_to_clean['path']))
            console.print(f"[success]✅ Deleted local data group '{dataset_to_clean['name']}'.[/success]")
        return True
    except Exception as e:
        console.print(f"[error]❌ Error deleting local data group: {e}[/error]")
        _LOGGER.exception(e)
        return False


def clean_all_datasets() -> bool:
    """Clean all top-level local cache namespaces and legacy dataset folders.

    Channel and tag-based resource subsets are intentionally excluded here
    because they overlap with the ``resources`` namespace.

    Returns:
        True if all local data was cleaned, False otherwise
    """
    datasets = discover_local_datasets()

    if not datasets:
        console.print("[dim]No local data groups found to clean.[/dim]")
        return True

    console.print(f"[warning]⚠️  About to delete {len(datasets)} local data group(s):[/warning]")

    total_size = 0
    for dataset in datasets:
        total_size += dataset['size_bytes']

    _render_local_data_groups_table(datasets, header_style="bold red", include_kind=True)
    console.print(f"\n[bold red]Total size to be deleted:[/bold red] {_format_size(total_size)}")
    console.print("[dim]This only removes local data and does not affect remote resources or annotations.[/dim]")

    confirmed = Confirm.ask(
        "Are you sure you want to delete ALL local data groups?",
        default=False,
        console=console,
    )

    if not confirmed:
        console.print("[dim]Operation cancelled.[/dim]")
        return False

    success_count = 0
    for dataset in datasets:
        try:
            resource_dirs = dataset.get('resource_dirs')
            if isinstance(resource_dirs, list):
                for rd in resource_dirs:
                    shutil.rmtree(rd)
                console.print(
                    f"[success]✅ Deleted {len(resource_dirs)} cached resource(s) for '{dataset['name']}'.[/success]"
                )
            else:
                shutil.rmtree(str(dataset['path']))
                console.print(f"[success]✅ Deleted: {dataset['name']}[/success]")
            success_count += 1
        except Exception as e:
            console.print(f"[error]❌ Failed to delete {dataset['name']}: {e}[/error]")
            _LOGGER.exception(e)

    if success_count == len(datasets):
        console.print(f"[success]✅ Successfully deleted all {success_count} local data groups.[/success]")
        return True
    else:
        console.print(
            f"[warning]⚠️  Deleted {success_count} out of {len(datasets)} local data groups.[/warning]"
        )
        return False


def interactive_dataset_cleaning() -> None:
    """Interactive local data cleaning menu."""
    base_groups = show_local_datasets()

    if not base_groups:
        return

    cleanable_groups = _discover_cleanable_local_data_groups(tag_limit=10)

    console.print("\n[title]🧹 Local Data Cleaning Options:[/title]")
    console.print(" [accent](1)[/accent] Clean a specific local data group")
    console.print(" [accent](2)[/accent] Clean all top-level local data groups")
    console.print(" [accent](b)[/accent] Back to main menu")

    try:
        choice = Prompt.ask("Enter your choice", console=console).lower().strip()

        # Handle ESC key (appears as escape sequence)
        if choice in ('', '\x1b', 'esc', 'escape'):
            return

        if choice == '1':
            console.print("\n[title]Available local data groups:[/title]")
            for i, dataset in enumerate(cleanable_groups, 1):
                console.print(
                    f" [accent]({i})[/accent] {dataset['name']} "
                    f"[dim]({dataset['kind']}, {dataset['item_count']} entries, {dataset['size']})[/dim]"
                )

            dataset_choice = Prompt.ask(
                "Enter group number, name, or selector (e.g. channel:training, tag:tutorial)",
                console=console,
            ).strip()

            # Handle ESC key in dataset selection
            if dataset_choice in ('', '\x1b', 'esc', 'escape'):
                return

            # Handle numeric choice
            try:
                dataset_idx = int(dataset_choice) - 1
                if 0 <= dataset_idx < len(cleanable_groups):
                    clean_dataset(str(cleanable_groups[dataset_idx]['identifier']))
                    return
            except ValueError:
                pass

            matched_group = _find_local_data_group(dataset_choice, cleanable_groups)
            if matched_group is None:
                console.print("[error]❌ Invalid local data group selection.[/error]")
            else:
                clean_dataset(str(matched_group['identifier']))

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
            console.print(" [accent](6)[/accent] Manage/Show local data...")
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
    console_handlers = [h for h in _USER_LOGGER.handlers if isinstance(h, ConsoleWrapperHandler)]
    if console_handlers:
        console = console_handlers[0].console
    parser = argparse.ArgumentParser(
        description='🔧 Datamint API Configuration Tool',
        epilog="""
Examples:
  datamint-config                           # Interactive mode
  datamint-config --api-key YOUR_KEY        # Set API key
    datamint-config --list-local-data         # Show local cache/data groups and filter selectors
  datamint-config --clean-local-data resources
                                           # Clean a cache namespace
    datamint-config --clean-local-data channel:training-data
                                                                                     # Clean cached resources for one upload channel
    datamint-config --clean-local-data tag:tutorial
                                                                                     # Clean cached resources matching one tag
  datamint-config --clean-local-data "Example Project"
                                           # Clean a legacy dataset folder
  datamint-config --clean-all-local-data    # Clean all local data groups
  
More Documentation: https://sonanceai.github.io/datamint-python-api/command_line_tools.html
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--api-key', type=str, help='API key to set')
    parser.add_argument('--default-url', '--url', type=str, help='Default URL to set')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode (default if no other arguments provided)')
    parser.add_argument(
        '--list-local-data', '--list-datasets',
        dest='list_local_data',
        action='store_true',
        help='List local cache namespaces, legacy dataset folders, and channel/tag selectors',
    )
    parser.add_argument(
        '--clean-local-data', '--clean-dataset',
        dest='clean_local_data',
        type=str,
        metavar='GROUP_NAME',
        help='Clean by cache namespace, legacy dataset, upload channel selector, or tag selector',
    )
    parser.add_argument(
        '--clean-all-local-data', '--clean-all-datasets',
        dest='clean_all_local_data',
        action='store_true',
        help='Clean all local cache namespaces and legacy dataset folders',
    )

    args = parser.parse_args()

    config_updates: dict[str, str] = {}

    if args.api_key is not None:
        config_updates[configs.APIKEY_KEY] = args.api_key

    if args.default_url is not None:
        # Basic URL validation
        if not (args.default_url.startswith('http://') or args.default_url.startswith('https://')):
            console.print("[error]❌ URL must start with http:// or https://[/error]")
            return
        config_updates[configs.APIURL_KEY] = args.default_url

    if config_updates:
        configs.set_values(config_updates)
        if args.api_key is not None:
            console.print("[success]✅ API key saved.[/success]")
        if args.default_url is not None:
            console.print("[success]✅ Default URL saved.[/success]")

    if args.list_local_data:
        show_local_datasets()

    if args.clean_local_data:
        clean_dataset(args.clean_local_data)

    if args.clean_all_local_data:
        clean_all_datasets()

    no_arguments_provided = (args.api_key is None and args.default_url is None and
                           not args.list_local_data and not args.clean_local_data and
                           not args.clean_all_local_data)

    if no_arguments_provided or args.interactive:
        interactive_mode()


if __name__ == "__main__":
    main()
