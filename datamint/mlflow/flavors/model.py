from datamint.entities.annotations import Annotation
from datamint.entities import Resource
from typing import Sequence, Any
from abc import ABC, abstractmethod
import mlflow
from mlflow.pyfunc import PyFuncModel, PythonModel


class DatamintModel(ABC, PythonModel):
    """
    Abstract adapter class for wrapping PyTorch models to produce Datamint annotations.

    Users should subclass this and implement the predict method to convert model outputs
    into Datamint Annotation objects.
    """

    def __init__(self, mlflow_models_uri: dict[str, str] = {}) -> None:
        super().__init__()
        self.mlflow_models_uri = mlflow_models_uri.copy()

    def load_context(self, context):
        self._mlflow_models = self._load_mlflow_models()

    def _load_mlflow_models(self):
        return {key: mlflow.pyfunc.load_model(uri)
                for key, uri in self.mlflow_models_uri.items()}

    @property
    def mlflow_models(self) -> dict[str, PyFuncModel]:
        if not hasattr(self, '_mlflow_models'):
            self._mlflow_models = self._load_mlflow_models()
        return self._mlflow_models

    @abstractmethod
    def predict(self,
                model_input: list[Resource],
                params: dict[str, Any] | None = None) -> Sequence[Sequence[Annotation]]:
        """
        Generate Datamint annotations for the given input data.

        Args:
            model_input: List of file paths or data identifiers
            params: Optional parameters for prediction

        Returns:
            List of annotation lists, one per input sample
        """
        pass
