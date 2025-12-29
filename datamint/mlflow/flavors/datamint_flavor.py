import mlflow
from mlflow.models import Model, ModelInputExample, ModelSignature
import datamint
import datamint.mlflow.flavors
from mlflow import pyfunc
from .model import DatamintModel
import logging
from collections.abc import Sequence
from dataclasses import asdict
from packaging.requirements import Requirement

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
               streamable=None,
               **kwargs):
    import medimgkit

    if mlflow_model is None:
        mlflow_model = Model()

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        datamint_version=datamint.__version__,
        supported_modes=supported_modes or datamint_model.get_supported_modes(),
        model_settings=asdict(datamint_model.settings),
        linked_models=datamint_model._get_linked_models_uri()
    )

    model_config = model_config or {}
    model_config.setdefault('device', 'cuda' if datamint_model.settings.need_gpu else 'cpu')
    
    def _get_req_name(req):
        try:
            return Requirement(req).name.lower()
        except Exception:
            return req.split("==")[0].strip().lower()

    datamint_requirements = ['datamint=={}'.format(datamint.__version__), 'medimgkit=={}'.format(medimgkit.__version__)]
    
    user_requirements = []
    # Check if requirements are lists (not strings which are also Sequences)
    if pip_requirements and isinstance(pip_requirements, Sequence) and not isinstance(pip_requirements, str):
        user_requirements.extend(pip_requirements)
    if extra_pip_requirements and isinstance(extra_pip_requirements, Sequence) and not isinstance(extra_pip_requirements, str):
        user_requirements.extend(extra_pip_requirements)
        
    user_req_names = {_get_req_name(req) for req in user_requirements}
    
    missing_requirements = [req for req in datamint_requirements if _get_req_name(req) not in user_req_names]

    if missing_requirements:
        if extra_pip_requirements is None:
            extra_pip_requirements = missing_requirements
        elif isinstance(extra_pip_requirements, Sequence) and not isinstance(extra_pip_requirements, str):
            extra_pip_requirements = list(extra_pip_requirements) + missing_requirements
        elif pip_requirements and isinstance(pip_requirements, Sequence) and not isinstance(pip_requirements, str):
             pip_requirements = list(pip_requirements) + missing_requirements


    datamint_model._clear_linked_models_cache()

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
        streamable=streamable,
        **kwargs
    )


def log_model(
    datamint_model: DatamintModel,
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
        datamint_model=datamint_model,
        supported_modes=supported_modes,
        name=name,
        flavor=datamint.mlflow.flavors.datamint_flavor,
        # loader_module=loader_module,
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
    if device is not None:
        model_config = {'device': device}
    else:
        model_config = None
    return mlflow.pyfunc.load_model(model_uri=model_uri,
                                    model_config=model_config
                                    ).unwrap_python_model()


def _load_pyfunc(path: str, model_config=None) -> pyfunc.PyFuncModel:
    return mlflow.pyfunc.load_model(model_uri=path, model_config=model_config)
