"""Command-line entry point for ``python -m datamint``."""

import argparse
import importlib
import importlib.metadata
import os
import sys

_COMMANDS: dict[str, str] = {
    "config": "datamint.client_cmd_tools.datamint_config",
    "upload": "datamint.client_cmd_tools.datamint_upload",
    "init": "datamint.client_cmd_tools.datamint_init",
    "train": "datamint.client_cmd_tools.datamint_train",
    "inference": "datamint.client_cmd_tools.datamint_inference",
    "example": "datamint.client_cmd_tools.datamint_example",
    "import": "datamint.client_cmd_tools.datamint_import",
}

_COMMAND_HELP: dict[str, str] = {
    "config": "Configure the API key, URL, and local cache",
    "upload": "Upload DICOM files and other resources",
    "init": "Generate starter scripts for a Datamint workflow",
    "train": "Train a model on a Datamint project",
    "inference": "Run local inference with a registered model",
    "example": "Populate a project with an example dataset",
    "import": "Import a labeled dataset (COCO, Pascal VOC, or YOLO)",
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


def _completing_command_hint() -> str | None:
    """Best-effort extraction of the subcommand token from COMP_LINE. """
    comp_line = os.environ.get("COMP_LINE", "")
    tokens = comp_line.split()
    if len(tokens) >= 2 and tokens[1] in _COMMANDS:
        return tokens[1]
    return None


def _autocomplete() -> None:
    """Build a combined parser tree and hand it to argcomplete. """
    import argcomplete

    parser = argparse.ArgumentParser(
        prog="datamint",
        description="Datamint command-line interface.",
        epilog=f"Available commands: {', '.join(_COMMANDS)}.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_resolve_version()}")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    target = _completing_command_hint()
    for name, module_path in _COMMANDS.items():
        if name != target:
            subparsers.add_parser(name, help=_COMMAND_HELP[name])
            continue
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            subparsers.add_parser(name, help=_COMMAND_HELP[name])
            continue
        build_fn = getattr(module, "_build_parser", None)
        if build_fn is not None:
            build_fn(subparsers)
        else:
            subparsers.add_parser(name, help=_COMMAND_HELP[name])

    argcomplete.autocomplete(parser)  


def main() -> None:
    if os.environ.get("_ARGCOMPLETE"):
        _autocomplete()
        return

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
