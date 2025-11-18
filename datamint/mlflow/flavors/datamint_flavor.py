import mlflow
from mlflow.models import Model, ModelInputExample, ModelSignature
import datamint
import datamint.mlflow.flavors
from mlflow import pyfunc
from .model import DatamintModel
import logging
from typing import Sequence
from dataclasses import asdict

FLAVOR_NAME = 'datamint'

_LOGGER = logging.getLogger(__name__)


def save_model(datamint_model: DatamintModel,
               path,
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
               example_no_conversion=None,
               streamable=None,
               **kwargs):
    if mlflow_model is None:
        mlflow_model = Model()

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        datamint_version=datamint.__version__,
        supported_modes=supported_modes or datamint_model.get_supported_modes(),
        model_settings=asdict(datamint_model.settings),
    )

    model_config = model_config or {}
    model_config.setdefault('device', 'cuda' if datamint_model.settings.need_gpu else 'cpu')

    return mlflow.pyfunc.save_model(
        path=path,
        python_model=datamint_model,
        data_path=data_path,
        conda_env=conda_env,
        mlflow_model=mlflow_model,
        # loader_module=None,
        artifacts=artifacts,
        code_paths=code_paths,
        infer_code_paths=infer_code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        model_config=model_config,
        example_no_conversion=example_no_conversion,
        streamable=streamable,
        **kwargs
    )


def log_model(
    datamint_model: DatamintModel,
    supported_modes: Sequence[str] | None = None,
    artifact_path: str = "datamint_model",
    data_path=None,
    code_paths=None,
    infer_code_paths=False,
    conda_env=None,
    artifacts=None,
    registered_model_name: str | None = None,
    signature: ModelSignature | None = None,
    input_example: ModelInputExample | None = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    model_config=None,
    example_no_conversion=None,
    streamable=None,
    **kwargs
):
    return Model.log(
        datamint_model=datamint_model,
        supported_modes=supported_modes,
        artifact_path=artifact_path,
        flavor=datamint.mlflow.flavors.datamint_flavor,
        # loader_module=loader_module,
        data_path=data_path,
        code_paths=code_paths,
        artifacts=artifacts,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        model_config=model_config,
        example_no_conversion=example_no_conversion,
        streamable=streamable,
        infer_code_paths=infer_code_paths,
        **kwargs
    )


def load_model(model_uri: str, device: str | None = None) -> DatamintModel:
    if device is not None:
        model_config = {'device': device}
    else:
        model_config = None
    return mlflow.pyfunc.load_model(model_uri=model_uri,
                                    model_config=model_config
                                    ).unwrap_python_model()


def _load_pyfunc(path: str, model_config=None) -> pyfunc.PyFuncModel:
    return mlflow.pyfunc.load_model(model_uri=path, model_config=model_config)
