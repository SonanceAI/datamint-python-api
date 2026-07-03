"""datamint-train command-line tool.

Train a model on a Datamint project using a built-in one-line trainer, without writing
any Python. Auto-detects the task (segmentation/classification/detection) and data format
(2D/3D) from the project's annotations and resources, picks a sensible default model, and
runs the full training pipeline.

Advanced tuning (custom losses, transforms, encoders, trainer_kwargs, etc.) is intentionally
not exposed here -- use the Python SDK (`datamint.lightning`) for that.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, TYPE_CHECKING

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from datamint import Api, configs
from datamint.client_cmd_tools.datamint_upload import handle_api_key
from datamint.exceptions import DatamintException
from datamint.utils.logging_utils import ConsoleWrapperHandler, load_cmdline_logging_config

if TYPE_CHECKING:
    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities import Project

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')
CONSOLE: Console

TASK_CHOICES = ('segmentation', 'classification', 'detection')

DEFAULT_MAX_EPOCHS = 5

# alias -> (task, Trainer class name in datamint.lightning)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    'unetpp':         ('segmentation', 'UNetPPTrainer'),
    'deeplabv3plus':  ('segmentation', 'DeepLabV3PlusTrainer'),
    'transunet':      ('segmentation', 'TransUNetTrainer'),
    'unetrpp':        ('segmentation', 'UNETRPPTrainer'),
    'nnunet':         ('segmentation', 'NNUNetTrainer'),
    'classification': ('classification', 'ImageClassificationTrainer'),
    'efficientnetv2': ('classification', 'EfficientNetV2Trainer'),
    'yolox':          ('detection', 'YOLOXTrainer'),
}

MODEL_FORMAT_RESTRICTION: dict[str, str] = {
    'unetrpp': '3d',
    'nnunet': '3d',
    'classification': '2d',
    'efficientnetv2': '2d',
    'yolox': '2d',
}

# Default model alias per (task, format) combination.
DEFAULT_MODEL_FOR: dict[tuple[str, str], str] = {
    ('segmentation', '2d'): 'unetpp',
    ('segmentation', '3d'): 'unetrpp',
    ('classification', '2d'): 'classification',
    ('detection', '2d'): 'yolox',
}

# Aliases whose own `image_size` default is None (no resize applied at all).
DEFAULT_IMAGE_SIZE_FOR: dict[str, int] = {
    'unetpp': 256,
    'deeplabv3plus': 256,
    'transunet': 224,  # TransUNetTrainer requires exactly (224, 224)
    'classification': 256,
}


class DatamintTrainCliError(Exception):
    """Expected, user-facing CLI error (bad input, ambiguous project, etc.)."""


def _detect(console: Console, project_name: str) -> tuple['DatamintBaseDataset', str, list[str]]:
    """Auto-detect data format and candidate task(s) for a project.

    Returns (dataset, format, tasks_present) where format is '2d' or '3d' and
    tasks_present is the list of task names whose annotations were found (usually one).
    """
    from datamint.dataset import ImageDataset, VideoDataset, VolumeDataset, build_dataset

    with console.status("[accent]Detecting task and data format...[/accent]"):
        try:
            dataset = build_dataset(project_name, allow_external_annotations=True)
        except ValueError as e:
            raise DatamintTrainCliError(str(e)) from e

    if isinstance(dataset, VideoDataset):
        raise DatamintTrainCliError(
            "Video projects aren't supported by datamint-train yet. "
            "Use the Python SDK (datamint.lightning) to build a custom training loop."
        )

    fmt = '2d' if isinstance(dataset, ImageDataset) else '3d'

    present = []
    if dataset.segmentation_labels_set:
        present.append('segmentation')
    if dataset.image_categories_set:
        present.append('classification')
    if dataset.box_labels_set:
        present.append('detection')

    return dataset, fmt, present


def _resolve_task(task_arg: str | None, present: list[str]) -> str:
    if task_arg is not None:
        return task_arg
    if len(present) == 0:
        raise DatamintTrainCliError(
            "Could not detect a task from this project's annotations (no segmentation, "
            "classification, or detection labels found). Pass --task explicitly."
        )
    if len(present) > 1:
        raise DatamintTrainCliError(
            f"This project has annotations for more than one task ({', '.join(present)}). "
            "Pass --task to pick one explicitly."
        )
    return present[0]


def _resolve_model_alias(model_arg: str | None, task: str, fmt: str) -> str:
    if model_arg is not None:
        model_task, _ = MODEL_REGISTRY[model_arg]
        if model_task != task:
            raise DatamintTrainCliError(
                f"--model {model_arg} trains a {model_task} model, but the task is {task}."
            )
        required_fmt = MODEL_FORMAT_RESTRICTION.get(model_arg)
        if required_fmt is not None and required_fmt != fmt:
            raise DatamintTrainCliError(
                f"--model {model_arg} only supports {required_fmt.upper()} projects, "
                f"but the detected data format is {fmt.upper()}."
            )
        return model_arg

    default_alias = DEFAULT_MODEL_FOR.get((task, fmt))
    if default_alias is None:
        raise DatamintTrainCliError(
            f"No built-in one-line trainer covers task='{task}' with format='{fmt}'. "
            "Use the Python SDK directly (see the Training your Model docs)."
        )
    return default_alias


def _get_trainer_class(alias: str):
    _, class_name = MODEL_REGISTRY[alias]
    try:
        import datamint.lightning as dl
    except ImportError as e:
        raise DatamintTrainCliError(
            f"Could not import Datamint's training dependencies ({e}). Make sure torch and "
            "lightning are installed, and if you're training with nnU-Net or a detection "
            "model, that the relevant extra is installed "
            "(pip install datamint[nnunet] / pip install datamint[detection])."
        ) from e
    return getattr(dl, class_name)


def _build_trainer_kwargs(args: argparse.Namespace, alias: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    if alias == 'nnunet':
        # nnU-Net manages its own batching/resizing/epoch semantics internally.
        if args.max_epochs is not None:
            kwargs['max_epochs'] = args.max_epochs
        kwargs['configuration'] = '2d'
    else:
        kwargs['max_epochs'] = args.max_epochs if args.max_epochs is not None else DEFAULT_MAX_EPOCHS
        if args.batch_size is not None:
            kwargs['batch_size'] = args.batch_size
        if alias != 'unetrpp':
            if args.image_size is not None:
                kwargs['image_size'] = args.image_size
            elif alias in DEFAULT_IMAGE_SIZE_FOR:
                kwargs['image_size'] = DEFAULT_IMAGE_SIZE_FOR[alias]

    if args.model_name is not None:
        kwargs['model_name'] = args.model_name

    return kwargs


def _print_plan(console: Console,
                project: 'Project',
                dataset: 'DatamintBaseDataset',
                fmt: str,
                task: str,
                model_alias: str,
                auto_picked: bool,
                kwargs: dict[str, Any]) -> None:
    table = Table(title="Training plan", show_header=False)
    table.add_column(style="key")
    table.add_column()
    table.add_row("Project", project.name)
    table.add_row("Resources", str(len(dataset)))
    table.add_row("Data format", fmt.upper())
    table.add_row("Task", task)
    table.add_row("Model", f"{model_alias} ({'auto-detected' if auto_picked else 'explicit'})")
    for key, value in kwargs.items():
        table.add_row(f"  {key}", str(value))
    console.print(table)

    if model_alias == 'nnunet':
        _print_nnunet_notice(console)


def _print_nnunet_notice(console: Console) -> None:
    console.print(
        "[warning]nnU-Net notice:[/warning] trains on fold 0 by default, and "
        "configuration defaults to '2d'. --batch-size and --image-size are ignored. "
        "See the notebooks or documentation for more details and advanced options."
    )


def _print_results(console: Console, trainer, results: dict[str, Any]) -> None:
    console.print()
    console.print("[success]✅ Training finished![/success]")

    test_results = results.get('test_results')
    if test_results:
        metrics = test_results[0] if isinstance(test_results, list) else test_results
        table = Table(title="Test metrics")
        table.add_column("Metric", style="key")
        table.add_column("Value")
        for k, v in metrics.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)

    console.print(f"MLflow experiment: [key]{trainer.experiment_name}[/key]")


def _resolve_project(api: Api, name: str) -> 'Project':
    project = api.projects.get_by_name(name)
    if project is not None:
        return project
    available = [p.name for p in api.projects.get_all()]
    raise DatamintTrainCliError(f"Project '{name}' not found. Available projects: {available}")


def _valid_aliases_for(task: str, fmt: str) -> list[str]:
    return sorted(
        alias for alias, (alias_task, _) in MODEL_REGISTRY.items()
        if alias_task == task and MODEL_FORMAT_RESTRICTION.get(alias, fmt) == fmt
    )


def _print_header(console: Console) -> None:
    console.print()
    console.rule("[bold]datamint-train[/bold]")
    console.print()
    console.print(" Answer a few questions and datamint-train will pick a model for you.")
    console.print()


def _run_interactive_wizard(console: Console, api: Api, args: argparse.Namespace) -> argparse.Namespace:
    _print_header(console)

    try:
        while True:
            project_name = Prompt.ask(" Project name", console=console).strip()
            if not project_name:
                console.print("[warning]Aborted.[/warning]")
                sys.exit(0)
            project = api.projects.get_by_name(project_name)
            if project is not None:
                break
            console.print(f"[error]Project '{project_name}' not found.[/error]")

        dataset, fmt, present = _detect(console, project.name)

        if len(present) == 1:
            task = present[0]
            console.print(f" Detected task: [key]{task}[/key] ({fmt.upper()} data)")
        else:
            console.print(f" Detected data format: [key]{fmt.upper()}[/key]")
            task = Prompt.ask(" Task", choices=list(TASK_CHOICES), console=console).strip()

        valid_aliases = _valid_aliases_for(task, fmt)
        if not valid_aliases:
            console.print(
                f"[error]No built-in one-line trainer covers task='{task}' with "
                f"format='{fmt.upper()}'. Use the Python SDK directly.[/error]"
            )
            sys.exit(1)

        default_alias = DEFAULT_MODEL_FOR.get((task, fmt))
        model_alias = Prompt.ask(" Model", choices=valid_aliases,
                                 default=default_alias, console=console).strip()

        epochs_default = 1000 if model_alias == 'nnunet' else DEFAULT_MAX_EPOCHS
        max_epochs = int(Prompt.ask(" Max epochs", default=str(epochs_default), console=console))

        args.project = project.name
        args.task = task
        args.model = model_alias
        args.max_epochs = max_epochs

        kwargs = _build_trainer_kwargs(args, model_alias)
        console.print()
        _print_plan(console, project, dataset, fmt, task, model_alias,
                   auto_picked=(model_alias == default_alias), kwargs=kwargs)

        if not Confirm.ask(" Start training?", default=True, console=console):
            sys.exit(0)
    except (KeyboardInterrupt, EOFError):
        console.print()
        sys.exit(0)

    return args


def _execute(args: argparse.Namespace, api: Api, console: Console, *, show_plan: bool = True) -> int:
    project = _resolve_project(api, args.project)
    dataset, fmt, present = _detect(console, project.name)
    task = _resolve_task(args.task, present)
    model_alias = _resolve_model_alias(args.model, task, fmt)
    auto_picked = args.model is None
    kwargs = _build_trainer_kwargs(args, model_alias)

    if show_plan:
        _print_plan(console, project, dataset, fmt, task, model_alias, auto_picked, kwargs)

    if args.dry_run:
        return 0

    trainer_cls = _get_trainer_class(model_alias)
    trainer = trainer_cls(project=project, **kwargs)

    console.print()
    console.print(f"[bold]Starting training with {trainer_cls.__name__}...[/bold]")
    results = trainer.fit()

    _print_results(console, trainer, results)

    if args.show_in_web:
        try:
            project.show()
        except Exception as e:
            console.print(f"[warning]Could not open browser: {e}. View it at: {project.url}[/warning]")

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a model on a Datamint project using a built-in one-line trainer.',
        epilog="""
Examples:
  datamint-train --project MyProject --model yolox --max-epochs 20
                                           # Train a specific model
  datamint-train --project MyProject      # Auto-detect task, data format, and model
  datamint-train --project MyProject --dry-run
                                           # Preview the detected plan without training
  datamint-train --interactive            # Guided wizard, no flags needed

More Documentation: https://sonanceai.github.io/datamint-python-api/command_line_tools.html#training-a-model
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--project', type=str, help='Project name to train on.')
    parser.add_argument('--model', type=str, choices=sorted(MODEL_REGISTRY),
                        help='Model to train: ' + ', '.join(sorted(MODEL_REGISTRY)) +
                        '. Auto-detected from the project if omitted.')
    parser.add_argument('--task', type=str, choices=list(TASK_CHOICES),
                        help='Task type. Auto-detected from the project annotations if omitted.')
    parser.add_argument('--max-epochs', type=int, default=None, help='Maximum training epochs.')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size. Ignored for --model nnunet.')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Target image size (square). Ignored for --model nnunet.')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name to register the trained model under in MLflow.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show the detected training plan without training.')
    parser.add_argument('--show-in-web', action='store_true',
                        help="Open the project's Datamint web page after training.")
    parser.add_argument('--interactive', action='store_true', help='Guided interactive wizard.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print debug messages.')

    args = parser.parse_args()

    if not args.interactive and not args.project:
        parser.error('--project is required (or use --interactive).')

    return args


def main() -> None:
    global CONSOLE
    load_cmdline_logging_config()
    CONSOLE = [h for h in _USER_LOGGER.handlers if isinstance(h, ConsoleWrapperHandler)][0].console

    args = _parse_args()

    if args.verbose:
        logging.getLogger().handlers[0].setLevel(logging.DEBUG)
        logging.getLogger('datamint').setLevel(logging.DEBUG)
        _LOGGER.setLevel(logging.DEBUG)
        _USER_LOGGER.setLevel(logging.DEBUG)

    try:
        api_key = handle_api_key()
        if api_key is None:
            _USER_LOGGER.error("API key not provided. Aborting.")
            sys.exit(1)
        os.environ[configs.ENV_VARS[configs.APIKEY_KEY]] = api_key

        try:
            api = Api(check_connection=True)
        except DatamintException as e:
            _USER_LOGGER.error(f'❌ Connection failed: {e}')
            sys.exit(1)

        if args.interactive:
            args = _run_interactive_wizard(CONSOLE, api, args)

        sys.exit(_execute(args, api, CONSOLE, show_plan=not args.interactive))
    except DatamintTrainCliError as e:
        _USER_LOGGER.error(f'❌ {e}')
        sys.exit(1)
    except DatamintException as e:
        _USER_LOGGER.error(f'❌ {e}')
        sys.exit(1)
    except KeyboardInterrupt:
        CONSOLE.print("\nTraining cancelled by user.", style='warning')
        sys.exit(1)


if __name__ == '__main__':
    main()
