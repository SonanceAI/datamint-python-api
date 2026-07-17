"""datamint inference command-line tool.

Run local inference with a registered Datamint model against a local file, without
writing any Python. Loads the model via MLflow (`models:/<name>/latest`), builds a
`LocalResource` from the given file, and prints the resulting predictions.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

from datamint.client_cmd_tools.datamint_upload import _is_valid_path_argparse, handle_api_key
from datamint.exceptions import DatamintException, ItemNotFoundError
from datamint.utils.env import is_legacy_cli_invocation
from datamint.utils.logging_utils import ConsoleWrapperHandler, load_cmdline_logging_config

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')
CONSOLE: Console

_OVERLAY_COLORS = ['red', 'lime', 'cyan', 'yellow', 'magenta', 'orange']


class DatamintInferenceCliError(Exception):
    """Expected, user-facing CLI error (bad input, model/project not found, etc.)."""


def _resolve_project_name(args: argparse.Namespace) -> str:
    return args.project if args.project is not None else args.model_name


def _load_model(project_name: str, model_name: str):
    from mlflow.exceptions import MlflowException

    from datamint.mlflow import set_project
    from datamint.mlflow.flavors import load_model

    try:
        set_project(project_name)
    except ItemNotFoundError as e:
        raise DatamintInferenceCliError(
            f"Project '{project_name}' not found. Models are looked up by project; if "
            f"'{model_name}' was registered under a different project name, pass --project "
            "explicitly."
        ) from e

    try:
        return load_model(f'models:/{model_name}/latest')
    except MlflowException as e:
        raise DatamintInferenceCliError(
            f"Could not load model '{model_name}' (project '{project_name}'): {e}"
        ) from e


def _run_inference(model, file_path: str, compute_uncertainty: bool = False):
    from datamint.entities.resource import LocalResource

    resource = LocalResource(local_filepath=file_path)
    params = {'compute_uncertainty': True} if compute_uncertainty else None
    predictions = model.predict([resource], params=params)
    return resource, (predictions[0] if predictions else [])


def _print_predictions(console: Console, predictions: list) -> None:
    if not predictions:
        console.print("[warning]No predictions returned.[/warning]")
        return

    table = Table(title="Predictions")
    table.add_column("Type", style="key")
    table.add_column("Identifier")
    table.add_column("Confidence")
    table.add_column("Uncertainty")
    for ann in predictions:
        confidence = getattr(ann, 'confiability', None)
        conf_str = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "-"
        uncertainty = getattr(ann, 'uncertainty', None)
        unc_str = f"{uncertainty:.3f}" if isinstance(uncertainty, (int, float)) else "-"
        identifier = getattr(ann, 'identifier', None) or getattr(ann, 'text_value', None) or '-'
        table.add_row(ann.annotation_type, str(identifier), conf_str, unc_str)
    console.print(table)


def _save_overlay(console: Console, resource: Any, predictions: list, output_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np

    img_np = np.array(resource.fetch_file_data(auto_convert=True))
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[0] != img_np.shape[-1]:
        img_np = np.transpose(img_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)

    drew_geometry = False
    labels_only: list[str] = []

    for i, ann in enumerate(predictions):
        color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]

        if ann.annotation_type == 'segmentation' and getattr(ann, 'mask', None) is not None:
            drew_geometry = True
            mask_np = np.asarray(ann.mask) > 0
            rgba = np.zeros((*mask_np.shape, 4))
            rgba[..., :3] = matplotlib.colors.to_rgb(color)
            rgba[..., 3] = 0.4 * mask_np
            ax.imshow(rgba)
        elif getattr(ann, 'geometry', None) is not None and hasattr(ann.geometry, 'points'):
            drew_geometry = True
            xs = [p[0] for p in ann.geometry.points]
            ys = [p[1] for p in ann.geometry.points]
            x_min, y_min = min(xs), min(ys)
            rect = patches.Rectangle(
                (x_min, y_min), max(xs) - x_min, max(ys) - y_min,
                linewidth=2, edgecolor=color, facecolor='none',
            )
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, ann.identifier or '', color=color, fontsize=9)
        else:
            label = getattr(ann, 'identifier', None) or getattr(ann, 'text_value', None)
            if label:
                labels_only.append(str(label))

    if labels_only:
        ax.set_title(', '.join(labels_only), fontsize=11)
    elif not drew_geometry:
        ax.set_title('No predictions', fontsize=11)

    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    console.print(f"[success]Saved visualization to {output_path}[/success]")


def _execute(args: argparse.Namespace, console: Console) -> int:
    project_name = _resolve_project_name(args)

    with console.status("[accent]Loading model...[/accent]"):
        model = _load_model(project_name, args.model_name)

    with console.status("[accent]Running inference...[/accent]"):
        resource, predictions = _run_inference(model, args.file, compute_uncertainty=args.uncertainty)

    _print_predictions(console, predictions)

    if args.output:
        _save_overlay(console, resource, predictions, args.output)

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run local inference with a registered Datamint model against a local file.',
        epilog="""
Examples:
  datamint inference file.png --model-name MyModel
                                           # Predict using the model registered as 'MyModel'
  datamint inference file.png --model-name my-model-alias --project MyProject
                                           # Model registered under a different name than its project
  datamint inference file.png --model-name MyModel --output result.png
                                           # Also save a visualization of the predictions
  datamint inference file.png --model-name MyModel --uncertainty
                                           # Also compute a predictive-uncertainty score

More Documentation: https://sonanceai.github.io/datamint-python-api/command_line_tools.html
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('file', type=_is_valid_path_argparse, metavar='FILE',
                        help='Path to the local file to run inference on.')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the registered model (loaded as models:/NAME/latest).')
    parser.add_argument('--project', type=str, default=None,
                        help="Project the model was registered under. Defaults to --model-name "
                        "itself (the default for every trainer unless model_name was set "
                        "explicitly to something else).")
    parser.add_argument('--output', type=str, default=None, metavar='PATH',
                        help='Save a rendered visualization of the predictions to this path.')
    parser.add_argument('--uncertainty', action='store_true', default=False,
                        help='Also compute a predictive-entropy uncertainty score per prediction '
                        '(see datamint.utils.uncertainty). Off by default.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print debug messages.')

    return parser.parse_args()


def main() -> None:
    global CONSOLE
    load_cmdline_logging_config()
    CONSOLE = [h for h in _USER_LOGGER.handlers if isinstance(h, ConsoleWrapperHandler)][0].console

    if is_legacy_cli_invocation('inference'):
        CONSOLE.print(
            "[warning]'datamint-inference' is deprecated and will be removed in a future "
            "release. Use 'datamint inference' instead.[/warning]"
        )

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
        import os

        from datamint import configs
        os.environ[configs.ENV_VARS[configs.APIKEY_KEY]] = api_key

        from datamint.mlflow.env_utils import ensure_mlflow_configured
        ensure_mlflow_configured()

        sys.exit(_execute(args, CONSOLE))
    except DatamintInferenceCliError as e:
        _USER_LOGGER.error(f'❌ {e}')
        sys.exit(1)
    except DatamintException as e:
        _USER_LOGGER.error(f'❌ {e}')
        sys.exit(1)
    except KeyboardInterrupt:
        CONSOLE.print("\nInference cancelled by user.", style='warning')
        sys.exit(1)


if __name__ == '__main__':
    main()
