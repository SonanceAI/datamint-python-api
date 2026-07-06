"""Command-line entry point for ``python -m datamint``."""

import argparse
import importlib
import importlib.metadata
import sys

_COMMANDS: dict[str, str] = {
    "config": "datamint.client_cmd_tools.datamint_config",
    "upload": "datamint.client_cmd_tools.datamint_upload",
    "init": "datamint.client_cmd_tools.datamint_init",
    "train": "datamint.client_cmd_tools.datamint_train",
    "inference": "datamint.client_cmd_tools.datamint_inference",
}


def _resolve_version() -> str:
    try:
        return importlib.metadata.version("datamint")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datamint",
        description="Datamint command-line interface.",
        epilog=f"Available commands: {', '.join(_COMMANDS)}.",
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
    argv = sys.argv[1:]

    if not argv:
        # Bare "datamint" with no subcommand: show help instead of an argparse error
        parser.print_help()
        sys.exit(0)

    # Parse only the first token (the command name) so that flags meant for the subcommand
    # (e.g. "datamint upload --help") are forwarded untouched instead of being swallowed by
    # this top-level parser's own -h/--help/--version handling, which would otherwise trigger
    # as soon as it sees them anywhere in the argument list.
    args = parser.parse_args(argv[:1])
    remaining = argv[1:]

    # Replace argv so the subcommand sees only its own arguments.
    # Note: a space (not a hyphen) so nested argparse usage lines read "datamint <command>"
    # (argparse's default prog is os.path.basename(sys.argv[0]), which is the string as-is
    # when it has no path separator).
    sys.argv = [f"datamint {args.command}", *remaining]

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
