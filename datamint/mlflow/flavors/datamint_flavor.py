import logging
from pathlib import Path
import mlflow
from mlflow.models import Model, ModelInputExample, ModelSignature
import datamint
import datamint.mlflow.flavors
from mlflow import pyfunc
from .model import BaseDatamintModel, DatamintModel, _DatamintModelWrapper
from .task_type import TaskType
from collections.abc import Sequence
from dataclasses import asdict
from packaging.requirements import Requirement
from typing import Any
import torch
import tempfile
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module

logger = logging.getLogger(__name__)

FLAVOR_NAME = 'datamint'
PYTORCH_DATA_SUBPATH = "pytorch_data"


def _process_signature(signature: ModelSignature | None,
                       python_model: BaseDatamintModel) -> ModelSignature:
    from mlflow.types import ParamSchema, ParamSpec
    from mlflow.models.signature import _infer_signature_from_type_hints

    # Define inference parameters
    params_schema = ParamSchema(
        [
            ParamSpec("mode", "string", "default"),  # Default mode
        ]
    )

    if signature is None:
        signature = _infer_signature_from_type_hints(
            python_model=python_model,
            context=None,
            type_hints=python_model.predict_type_hints,
            input_example=None,
        )
    assert signature is not None

    # Merge existing params with our new params, ensuring no duplicates
    existing_params: list[ParamSpec] = signature.params.params if signature.params else []
    existing_param_names = {param.name for param in existing_params}
    new_params = [param for param in params_schema.params if param.name not in existing_param_names]
    signature.params = ParamSchema(existing_params + new_params)

    return signature


def _process_input_example(input_example: ModelInputExample | None) -> tuple[ModelInputExample | None, dict[str, Any]]:

    logger.info('Processing input example is disabled for now')
    raise NotImplementedError('Processing input example is disabled for now')

    datamint_params = {
        "mode": "default",
    }
    if input_example is None:
        from datamint.entities.resource import LocalResource
        input_resource = LocalResource(raw_data=bytes()).model_dump(mode='json')
        return [input_resource], datamint_params
    if not isinstance(input_example, tuple):
        return (input_example, datamint_params)
    data_example, params_example = input_example
    merged_params = {**(params_example or {}), **datamint_params}
    return (data_example, merged_params)


# Path where MLflow copies the model in the deploy container
_DEPLOY_MODEL_PATH = '/opt/ml/model'


def _datamint_requirement() -> tuple[str, Path | None]:
    """Return (requirement_string, local_wheel_path_or_None).

    When datamint is installed in editable mode the wheel is built locally so
    it can be bundled directly into the model directory.  The requirement string
    then references the ``file://`` path it will have inside the deploy container
    after ``COPY model_dir/ /opt/ml/model``.
    """
    import importlib.metadata as _imeta
    import json as _json

    dist = _imeta.Distribution.from_name('datamint')
    direct_url_text = dist.read_text('direct_url.json')

    if direct_url_text is None:
        return f'datamint=={datamint.__version__}', None

    info = _json.loads(direct_url_text)
    if not info.get('dir_info', {}).get('editable', False):
        return f'datamint=={datamint.__version__}', None

    from urllib.parse import urlparse
    source_dir = Path(urlparse(info['url']).path)
    return _build_datamint_wheel(source_dir)


def _build_datamint_wheel(source_dir: Path) -> tuple[str, Path]:
    """Build a wheel from *source_dir* and return (requirement_str, wheel_path).

    The wheel is written to a mkdtemp directory; the caller is responsible for
    cleanup (done in save_model via try/finally).

    Venvs inside the package dir (e.g. datamint/env/) contain symlinks that
    resolve to system paths outside the project root, causing poetry-core to
    crash with a ValueError.  Building from a clean copy avoids this.
    """
    import sys
    import subprocess
    import shutil
    import tempfile as _tmp

    _IGNORE = shutil.ignore_patterns(
        'env', '.venv', 'venv',
        '*.egg-info', '__pycache__', '.git', 'dist', 'build', '.tox',
    )

    logger.info("Building datamint wheel from %s for deploy container...", source_dir)
    tmpdir = Path(_tmp.mkdtemp(prefix='datamint_wheel_'))
    build_src = tmpdir / 'src'
    wheel_dir = tmpdir / 'wheel'
    wheel_dir.mkdir(parents=True)

    try:
        shutil.copytree(source_dir, build_src, ignore=_IGNORE, symlinks=False)
        subprocess.run(
            [sys.executable, '-m', 'pip', 'wheel', str(build_src), '--no-deps', '-w', str(wheel_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(
            f"pip wheel failed for {source_dir}.\n"
            f"stdout:\n{e.output}\nstderr:\n{e.stderr}"
        ) from e

    wheels = list(wheel_dir.glob('datamint*.whl'))
    if not wheels:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"pip wheel produced no datamint wheel in {wheel_dir}.")

    wheel_path = wheels[0]
    requirement = f'datamint @ file://{_DEPLOY_MODEL_PATH}/artifacts/{wheel_path.name}'
    logger.info("datamint wheel built at %s; will be bundled with model", wheel_path)
    return requirement, wheel_path


def _resolve_requirements(
    pip_requirements,
    extra_pip_requirements,
) -> tuple[object, object, dict[str, str]]:
    """Return (pip_requirements, extra_pip_requirements, extra_artifacts).

    *extra_artifacts* maps artifact key -> local file path for any wheel files
    that must be copied into the model directory so pip can install them without
    an MLflow client inside the Docker build.
    """
    import medimgkit

    def _get_req_name(req):
        if req.endswith('.whl'):
            return req.split('/')[-1].split('-')[0].lower().replace('_', '-')
        try:
            return Requirement(req).name.lower()
        except Exception:
            return req.split("==")[0].strip().lower()

    datamint_req, datamint_wheel_path = _datamint_requirement()

    datamint_requirements = [
        datamint_req,
        f'medimgkit=={medimgkit.__version__}',
    ]

    user_requirements = []
    if isinstance(pip_requirements, Sequence) and not isinstance(pip_requirements, str):
        user_requirements.extend(pip_requirements)
    if isinstance(extra_pip_requirements, Sequence) and not isinstance(extra_pip_requirements, str):
        user_requirements.extend(extra_pip_requirements)

    user_req_names = {_get_req_name(req) for req in user_requirements}
    missing_requirements = [req for req in datamint_requirements if _get_req_name(req) not in user_req_names]

    if missing_requirements:
        if extra_pip_requirements is None:
            extra_pip_requirements = missing_requirements
        elif isinstance(extra_pip_requirements, Sequence) and not isinstance(extra_pip_requirements, str):
            extra_pip_requirements = list(extra_pip_requirements) + missing_requirements
        elif isinstance(pip_requirements, Sequence) and not isinstance(pip_requirements, str):
            pip_requirements = list(pip_requirements) + missing_requirements

    extra_artifacts: dict[str, str] = {}
    if datamint_wheel_path is not None:
        extra_artifacts['datamint_wheel'] = str(datamint_wheel_path)

    return pip_requirements, extra_pip_requirements, extra_artifacts


def save_model(datamint_model: BaseDatamintModel,
               path,
               task_type: TaskType | str | None = None,
               supported_modes: Sequence[str] | None = None,
               data_path=None,
               code_paths=None,
               infer_code_paths=False,
               conda_env=None,
               mlflow_model: Model | None = None,
               artifacts=None,
               signature: ModelSignature | None = None,
               input_example: ModelInputExample | None = None,
               pip_requirements=None,
               extra_pip_requirements=None,
               metadata=None,
               model_config=None,
               streamable=None,
               **kwargs):
    logger.debug(f"Saving DatamintModel to path: {path} "
                 f" class name: {datamint_model.__class__.__name__} "
                 f' has_input_example: {input_example is not None} ')

    if not isinstance(datamint_model, DatamintModel):
        datamint_model = _DatamintModelWrapper(datamint_model)

    if mlflow_model is None:
        mlflow_model = Model()

    model_config = model_config or {}
    model_config.setdefault('device', 'cuda' if datamint_model.settings.need_gpu else 'cpu')

    pip_requirements, extra_pip_requirements, extra_artifacts = _resolve_requirements(
        pip_requirements, extra_pip_requirements
    )
    # Merge any bundled wheels (e.g. editable-install datamint) into the model
    # artifacts dict so MLflow copies them into the model directory.
    artifacts = {**(artifacts or {}), **extra_artifacts}
    # Temp dirs holding built wheels; cleaned up after pyfunc.save_model copies them.
    _wheel_tmpdirs = {Path(p).parent.parent for p in extra_artifacts.values()}

    if hasattr(datamint_model, '_clear_linked_models_cache'):
        datamint_model._clear_linked_models_cache()

    if signature is not None:
        signature = _process_signature(signature, datamint_model)
    try:
        input_example = _process_input_example(input_example)
    except NotImplementedError:
        input_example = None
    except Exception as e:
        logger.warning(f"Failed to process input example. Proceeding without input example. Error: {e}")
        input_example = None

    linked_models = datamint_model._get_linked_models_uri() if hasattr(datamint_model, '_get_linked_models_uri') else {}
    resolved_task_type = task_type or getattr(datamint_model, 'task_type', None)
    task_type_value = resolved_task_type.value if isinstance(resolved_task_type, TaskType) else resolved_task_type
    flavor_params = {
        "datamint_version": datamint.__version__,
        "supported_modes": supported_modes or datamint_model.get_supported_modes(),
        "model_settings": asdict(datamint_model.settings),
        "linked_models": linked_models,
        "task_type": task_type_value,
    }
    mlflow_model.add_flavor(FLAVOR_NAME, **flavor_params)
    model_config.update(flavor_params)

    pyfunc_kwargs = dict(
        path=path,
        python_model=datamint_model,
        data_path=data_path,
        conda_env=conda_env,
        mlflow_model=mlflow_model,
        artifacts=artifacts,
        code_paths=code_paths,
        infer_code_paths=infer_code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        model_config=model_config,
        streamable=streamable,
        **kwargs
    )

    import shutil as _shutil

    pt_model = datamint_model.get_pytorch_model() if hasattr(datamint_model, 'get_pytorch_model') else None
    if pt_model is not None:
        if hasattr(datamint_model, '_clear_linked_models_cache'):
            datamint_model._clear_linked_models_cache()
        datamint_model._clear_ptmodel()
        with tempfile.NamedTemporaryFile() as tmp_file:
            logger.debug(f"Saving PyTorch model to temporary file {tmp_file.name}")
            torch.save(pt_model, tmp_file.name, pickle_module=mlflow_pytorch_pickle_module)
            pyfunc_kwargs['artifacts'] = {
                **artifacts,
                DatamintModel._PYTORCH_ARTIFACT_NAME: tmp_file.name,
            }
            logger.debug(
                "Saving PyFunc model with PyTorch artifact for model %s...",
                datamint_model.__class__.__name__,
            )
            try:
                return mlflow.pyfunc.save_model(**pyfunc_kwargs)
            finally:
                for d in _wheel_tmpdirs:
                    _shutil.rmtree(d, ignore_errors=True)

    # DatamintLightningModule is an nn.Module itself, so its CUDA weights are
    # embedded directly in the cloudpickle. Move to CPU before serialization
    # so the pickle is device-agnostic and loads on CPU-only containers.
    import torch.nn as nn
    _underlying = datamint_model.another_model if isinstance(datamint_model, _DatamintModelWrapper) else datamint_model
    if isinstance(_underlying, nn.Module):
        _underlying.cpu()

    logger.debug(f'Saving PyFunc model for model {datamint_model.__class__.__name__}...')
    try:
        return mlflow.pyfunc.save_model(**pyfunc_kwargs)
    finally:
        for d in _wheel_tmpdirs:
            _shutil.rmtree(d, ignore_errors=True)


def log_model(
    datamint_model: BaseDatamintModel,
    task_type: TaskType | str | None = None,
    supported_modes: Sequence[str] | None = None,
    name: str = "datamint_model",
    data_path=None,
    code_paths=None,
    infer_code_paths=False,
    artifacts=None,
    registered_model_name: str | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample | None = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    model_config=None,
    **kwargs
):
    return Model.log(
        artifact_path=None,
        datamint_model=datamint_model,
        task_type=task_type,
        supported_modes=supported_modes,
        name=name,
        flavor=datamint.mlflow.flavors.datamint_flavor,
        data_path=data_path,
        code_paths=code_paths,
        artifacts=artifacts,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        model_config=model_config,
        infer_code_paths=infer_code_paths,
        **kwargs
    )


def load_model(model_uri: str, device: str | None = None) -> DatamintModel:
    model_config = {'device': device} if device is not None else None

    from mlflow.tracking.artifact_utils import _download_artifact_from_uri
    local_path = _download_artifact_from_uri(artifact_uri=model_uri)

    return _load_pyfunc(local_path, model_config=model_config).unwrap_python_model()


def _load_pyfunc(path: str, model_config=None) -> pyfunc.PyFuncModel:
    logger.debug(f"Loading PyFunc model from path: {path} with model_config: {model_config}")
    pf_model = mlflow.pyfunc.load_model(model_uri=path, model_config=model_config)
    dt_model = pf_model.unwrap_python_model()
    if isinstance(dt_model, _DatamintModelWrapper):
        logger.debug("Unwrapping DatamintModel from wrapper")
        dt_model = dt_model.another_model
        pf_model._model_impl.python_model = dt_model

    # Restore task_type from flavor metadata if not already set on the model
    if not dt_model.task_type:
        try:
            mlflow_model_meta = Model.load(path)
            flavor_data = mlflow_model_meta.flavors.get(FLAVOR_NAME, {})
            task_type_str = flavor_data.get('task_type')
            if task_type_str:
                dt_model.task_type = TaskType(task_type_str)
        except ValueError:
            logger.warning(f"Unknown task_type in flavor metadata: {task_type_str}")
        except Exception as e:
            logger.debug(f"Could not restore task_type from flavor metadata: {e}")

    return pf_model
