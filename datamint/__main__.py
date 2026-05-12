"""Command-line entry point for ``python -m datamint``."""

import argparse
import importlib
import importlib.metadata
import sys

_COMMANDS: dict[str, str] = {
    "config": "datamint.client_cmd_tools.datamint_config",
    "upload": "datamint.client_cmd_tools.datamint_upload",
}


def _resolve_version() -> str:
    try:
        return importlib.metadata.version("datamint")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m datamint",
        description="Datamint command-line interface.",
        epilog="Available commands: config, upload",
    )
    parser.add_argument(
        "command",
        choices=_COMMANDS,
        metavar="command",
        help=f"Subcommand to run. Choices: {', '.join(_COMMANDS)}.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_resolve_version()}",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    # Parse only the first positional argument; leave the rest for the subcommand.
    args, remaining = parser.parse_known_args()

    # Replace argv so the subcommand sees only its own arguments.
    sys.argv = [f"datamint-{args.command}", *remaining]

    module_path = _COMMANDS[args.command]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        parser.error(f"Failed to import module for command '{args.command}': {exc}")

    entry_point = getattr(module, "main", None)
    if entry_point is None:
        parser.error(f"Module '{module_path}' does not expose a 'main()' function.")

    entry_point()


if __name__ == "__main__":
    main()
