#!/usr/bin/env python3
"""
MLflow model serving script with MLServer command patching.
"""

import os
from mlflow.models import container as C
from mlflow.pyfunc import mlserver
import logging

_LOGGER = logging.getLogger(__name__)


def _configure_gpu_device():
    """Configure GPU device based on environment variables.

    Sets ``CUDA_VISIBLE_DEVICES`` if a GPU is requested via
    ``MLFLOW_DEFAULT_PREDICTION_DEVICE`` but not already configured.
    Also attempts to detect CUDA availability at startup.
    """
    device = os.environ.get('MLFLOW_DEFAULT_PREDICTION_DEVICE', 'cpu')
    if device == 'cuda':
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            _LOGGER.info(
                "CUDA_VISIBLE_DEVICES already set to: %s",
                os.environ['CUDA_VISIBLE_DEVICES'],
            )
        else:
            # Only set CUDA_VISIBLE_DEVICES when torch reports a concrete
            # device count. An empty value (0 GPUs) would hide every GPU,
            # and a hardcoded '0' would restrict non-torch frameworks to a
            # single GPU. Leaving it unset exposes all GPUs to the runtime.
            try:
                import torch
                num_gpus = torch.cuda.device_count()
            except ImportError:
                num_gpus = None
            if num_gpus:
                gpu_indices = ','.join(str(i) for i in range(num_gpus))
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_indices
                _LOGGER.info(
                    "Set CUDA_VISIBLE_DEVICES=%s (%d GPU(s) available)",
                    gpu_indices, num_gpus,
                )
            elif num_gpus == 0:
                _LOGGER.warning(
                    "MLFLOW_DEFAULT_PREDICTION_DEVICE=cuda but torch reports 0 GPUs. "
                    "Leaving CUDA_VISIBLE_DEVICES unset; model will likely fall back to CPU."
                )
            else:
                _LOGGER.warning(
                    "PyTorch not installed; leaving CUDA_VISIBLE_DEVICES unset "
                    "(all GPUs visible to the runtime)"
                )

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                num_gpus = torch.cuda.device_count()
                _LOGGER.info(
                    "CUDA available: %s (torch.cuda.device_count=%d)",
                    gpu_name, num_gpus,
                )
            else:
                _LOGGER.warning(
                    "MLFLOW_DEFAULT_PREDICTION_DEVICE=cuda but torch.cuda.is_available()=False. "
                    "CUDA_VISIBLE_DEVICES=%s. Model will likely fall back to CPU.",
                    os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
                )
        except ImportError:
            _LOGGER.info("PyTorch not installed; skipping CUDA availability check")
    else:
        # Explicitly clear CUDA_VISIBLE_DEVICES for CPU-only mode
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        _LOGGER.info("Running on device: %s", device)


def _patch_get_cmd():
    """Patch mlserver.get_cmd to print the command before returning it and fix a bug in mlflow"""
    original_get_cmd = mlserver.get_cmd

    def patched_get_cmd(*args, **kwargs):
        cmd, env_vars = original_get_cmd(*args, **kwargs)

        for key, value in env_vars.items():
            if isinstance(value, (int, float, bool)):
                env_vars[key] = str(value)

        _LOGGER.info(f'MLServer command: {cmd}')
        _LOGGER.info(f'MLServer environment variables: {env_vars}')
        return cmd, env_vars

    mlserver.get_cmd = patched_get_cmd


def main():
    _patch_get_cmd()
    _configure_gpu_device()
    C._serve('local')


if __name__ == '__main__':
    import rich.logging

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(rich.logging.RichHandler())

    main()
