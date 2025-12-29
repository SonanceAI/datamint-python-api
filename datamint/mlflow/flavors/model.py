"""
DataMint Model Adapter Module

This module provides a flexible framework for wrapping ML models to work with DataMint's
annotation system. It supports various prediction modes for different data types and use cases.
"""

from typing import Any, TypeAlias
from collections.abc import Callable
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from mlflow.environment_variables import MLFLOW_DEFAULT_PREDICTION_DEVICE
from mlflow.pyfunc import load_model as pyfunc_load_model
from mlflow.pytorch import load_model as pytorch_load_model
from mlflow.pyfunc import PyFuncModel, PythonModel, PythonModelContext
from datamint.entities.annotations import Annotation
from datamint.entities.resource import Resource
import logging
import os

logger = logging.getLogger(__name__)

# Type aliases
PredictionResult: TypeAlias = list[list[Annotation]]


@dataclass
class ModelSettings:
    """
    Deployment and inference configuration for DatamintModel.

    These settings are serialized with the model and used by remote MLflow servers
    to properly configure the runtime environment.
    """
    # Hardware requirements
    need_gpu: bool = False
    """Whether GPU is required for inference"""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ModelSettings':
        """Create config from dictionary, raising error on unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        invalid_fields = set(data.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(f"Invalid fields for ModelSettings: {', '.join(sorted(invalid_fields))}")
        return cls(**data)


class PredictionMode(str, Enum):
    """
    Enumeration of supported prediction modes.

    Each mode corresponds to a specific method signature in DatamintModel.
    """
    # Standard modes
    DEFAULT = 'default'                # Default: process entire resource as-is

    # Simple modes
    IMAGE = 'image'                    # Process single 2d image resource

    # Video/temporal modes
    FRAME = 'frame'                    # Extract and process specific frame
    FRAME_RANGE = 'frame_range'        # Process contiguous frame range
    ALL_FRAMES = 'all_frames'          # Process all frames independently
    TEMPORAL_SEQUENCE = 'temporal_sequence'  # Process with temporal context window

    # 3D volume modes
    SLICE = 'slice'                    # Extract and process specific slice
    SLICE_RANGE = 'slice_range'        # Process contiguous slice range
    PRIMARY_SLICE = 'primary_slice'    # Process center/primary slice
    # MULTI_PLANE = 'multi_plane'        # Process multiple anatomical planes
    VOLUME = 'volume'                  # Process entire 3D volume

    # Spatial modes
    # ROI = 'roi'                        # Process single region of interest
    # MULTI_ROI = 'multi_roi'            # Process multiple regions
    # TILE = 'tile'                      # Split into tiles (whole slide imaging)
    # PATCH = 'patch'                    # Extract patches around points

    # Advanced modes
    INTERACTIVE = 'interactive'        # With user prompts (SAM-like)
    FEW_SHOT = 'few_shot'             # With context examples
    # MULTI_VIEW = 'multi_view'          # Multiple views of same subject


class DatamintModel(ABC, PythonModel):
    """
    Abstract adapter class for wrapping models to produce Datamint annotations.

    This class provides a flexible framework for integrating ML models with DataMint.
    The main `predict()` method routes requests to specific handlers based on the
    prediction mode, allowing users to implement only the modes they need.

    Quick Start:
    -----------
    ```python
    class MyModel(DatamintModel):
        def __init__(self):
            super().__init__(
                mlflow_models_uri={'model': 'models:/MyModel/latest'},
                config=ModelSettings(need_gpu=True)
            )

        def predict_default(self, model_input, **kwargs):
            # Access the device for your computation
            device = self.inference_device  # Reads from MLFLOW_DEFAULT_PREDICTION_DEVICE or defaults to 'cpu'
            model = self.mlflow_models['model'].get_raw_model().to(device)
            # ... process and return annotations
            return predictions
    ```

    Prediction Modes:
    ----------------
    Users can request different prediction modes via params['mode']:

    **Default**: Default processing
    ```python
    model.predict(resources)  # or params={'mode': 'default'}
    ```

    **Video Frame**: Extract specific frame
    ```python
    model.predict(videos, params={'mode': 'frame', 'frame_index': 42})
    ```

    **3D Slice**: Extract specific slice
    ```python
    model.predict(volumes, params={'mode': 'slice', 'slice_index': 50, 'axis': 'axial'})
    ```

    **Interactive**: With prompts
    ```python
    model.predict(images, params={'mode': 'interactive', 'prompt': {'points': [[x, y]], 'labels': [1]}})
    ```

    Common Parameters:
    -----------------
    - `confidence_threshold` (float): Filter predictions by confidence score
    - `batch_size` (int): Batch size for processing
    - `render_annotation` (bool): Return annotated images instead of annotations

    Device Configuration:
    --------------------
    The device for computation is automatically configured from the 
    `MLFLOW_DEFAULT_PREDICTION_DEVICE` environment variable. Access it via `self.inference_device`.
    Defaults to 'cpu' if not set.

    Implementation Guide:
    --------------------
    1. Implement `predict_default()` - this is required and serves as fallback
    2. Optionally implement specific modes your model supports
    3. Override `_render_annotations()` if you want to support visualization
    4. Use `self.mlflow_models` to access loaded MLflow models
    5. Configure deployment settings via `ModelSettings`

    See individual method docstrings for detailed parameter specifications.
    """

    LINKED_MODELS_DIR = "linked_models"
    _CACHED_ATTRS = ['_mlflow_models', '_mlflow_torch_models', '_inference_device']

    def __init__(self,
                 settings: ModelSettings | dict[str, Any] | None = None,
                 mlflow_torch_models_uri: dict[str, str] | None = None,
                 mlflow_models_uri: dict[str, str] | None = None,
                 ) -> None:
        """
        Initialize the DatamintModel adapter.

        Args:
            config: ModelSettings instance or dict with deployment settings.
                    Example: {'need_gpu': True}
            mlflow_torch_models_uri: Dictionary mapping model names to PyTorch model URIs.
                    Example: {'backbone': 'models:/MyClassifier/2'}
                    These models will be lazy-loaded and accessible via ``self.mlflow_torch_models_uri['backbone']``
            mlflow_models_uri: Dictionary mapping model names to MLflow URIs.
                    Example: {'detector': 'models:/MyDetector/1',
                              'classifier': 'models:/MyClassifier/latest'}
                    These models will be lazy-loaded and accessible via ``self.mlflow_models['detector']``

        """
        super().__init__()
        self.mlflow_models_uri = (mlflow_models_uri or {}).copy()
        self.mlflow_torch_models_uri = (mlflow_torch_models_uri or {}).copy()

        # Handle settings - convert dict to ModelSettings if needed
        if isinstance(settings, dict):
            self.settings = ModelSettings.from_dict(settings)
        elif isinstance(settings, ModelSettings):
            self.settings = settings
        else:
            self.settings = ModelSettings()

        self._supported_modes_cache = None

    def load_context(self, context: PythonModelContext):
        """
        Called by MLflow when loading the model.

        Override this if you need custom loading logic.
        """
        self._inference_device = self._load_inference_device(context=context)
        self._mlflow_models = self._load_mlflow_models()
        self._mlflow_torch_models = self._load_mlflow_torch_models()
        # model_config = context.model_config

    def _get_linked_models_uri(self) -> dict[str, Any]:
        """Get all linked models (MLflow and PyTorch)"""
        linked = {}
        linked.update(self.mlflow_models_uri)
        linked.update(self.mlflow_torch_models_uri)
        return linked

    def _clear_linked_models_cache(self):
        """Clear loaded linked models to free memory"""
        
        for attr in self._CACHED_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)
        

    def __getstate__(self):
        state = self.__dict__.copy()

        for attr in self._CACHED_ATTRS:
            if attr in state:
                del state[attr]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # avoid possible invalid states after unpickling
        self._clear_linked_models_cache()

    def _load_inference_device(self, context: PythonModelContext | None = None) -> str:
        """
        Load inference device from model config or environment variable.
        """
        import torch

        device = None
        if context and context.model_config:
            device = context.model_config.get("device", None)
            logger.info(f"Model config device: {device}")
        if device is None:
            env_device = MLFLOW_DEFAULT_PREDICTION_DEVICE.get()
            if env_device:
                device = env_device
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        logger.info(f"Set inference device: {device}")
        return device

    @property
    def inference_device(self) -> str:
        if hasattr(self, '_inference_device') and self._inference_device is not None:
            return self._inference_device
        env_device = MLFLOW_DEFAULT_PREDICTION_DEVICE.get()
        if env_device:
            logger.info(f"Inference device not set; getting from environment variable ({env_device})")
            return env_device
        logger.warning("Inference device not set; defaulting to 'cpu'")
        return 'cpu'

    def _load_models_generic(self, uris: dict[str, str],
                             loader_func: Callable,
                             **loader_kwargs) -> dict[str, Any]:
        """Generic helper to load models from URIs."""
        loaded_models = {}
        for name, uri in uris.items():
            model_uri = uri
            if os.path.exists(uri):
                logger.info(f"Model '{name}' found locally at '{uri}'")
                model_uri = os.path.abspath(uri)
            elif uri.startswith("models:/"):
                local_path = uri.replace("models:/", DatamintModel.LINKED_MODELS_DIR + "/", 1)
                if os.path.exists(local_path):
                    logger.info(f"Model '{name}' found locally at '{local_path}'")
                    model_uri = os.path.abspath(local_path)

            try:
                loaded_models[name] = loader_func(model_uri, **loader_kwargs)
                logger.info(f"Loaded model '{name}' from {model_uri}")
            except Exception as e:
                logger.error(f"Failed to load model '{name}' from {model_uri}: {e}")
                raise
        return loaded_models

    def _load_mlflow_models(self) -> dict[str, PyFuncModel]:
        """Load all MLflow models specified in mlflow_models_uri."""
        return self._load_models_generic(
            self.mlflow_models_uri,
            pyfunc_load_model,
            model_config={'device': self.inference_device}
        )

    def _load_mlflow_torch_models(self) -> dict[str, Any]:
        """Load all MLflow PyTorch models specified in mlflow_torch_models_uri."""
        models = self._load_models_generic(
            self.mlflow_torch_models_uri,
            pytorch_load_model,
            device=self.inference_device,
            map_location=self.inference_device,
        )
        for m in models.values():
            if hasattr(m, 'eval'):
                m.eval()
        return models

    def get_mlflow_models(self) -> dict[str, PyFuncModel]:
        """
        Access loaded MLflow models.

        Returns:
            Dictionary mapping model names to PyFuncModel instances.
            Use .get_raw_model() to access the underlying model (e.g., torch.nn.Module)
        """
        if not hasattr(self, '_mlflow_models'):
            logger.warning("Loading MLflow models on first access")
            self._mlflow_models = self._load_mlflow_models()
        return self._mlflow_models

    def get_mlflow_torch_models(self) -> dict[str, Any]:
        """
        Access loaded MLflow PyTorch models.

        Returns:
            Dictionary mapping model names to PyTorch model instances.
        """
        if not hasattr(self, '_mlflow_torch_models'):
            logger.warning("Loading MLflow PyTorch models on first access")
            self._mlflow_torch_models = self._load_mlflow_torch_models()
        return self._mlflow_torch_models

    # def _preprocess_input(self,
    #                       model_input: list[InferenceResource | Resource | dict[str, Any]],
    #                       params: dict[str, Any]) -> list[Resource]:
    #     """
    #     Preprocess input to convert to list of Resource objects.

    #     Args:
    #         model_input: List of InferenceResource, Resource, or dict
    #         params: Additional parameters (unused here)
    #     Returns:
    #         List of Resource objects
    #     """
    #     resources = []
    #     for item in model_input:
    #         if isinstance(item, Resource):
    #             resources.append(item)
    #         elif isinstance(item, InferenceResource):
    #             resources.append(item.fabricate_resource())
    #         elif isinstance(item, dict):
    #             if 'local_filepath' in item or item.get('id', None) == '':
    #                 logger.debug(f'Creating LocalResource from dict: {item}')
    #                 resources.append(LocalResource(local_filepath=item['local_filepath']))
    #             elif 'upload_channel' in item or 'location' in item or 'storage' in item:
    #                 resources.append(Resource(**item))
    #             else:
    #                 resources.append(InferenceResource(**item).fabricate_resource())
    #         else:
    #             raise ValueError(f"Unsupported input type: {type(item)}")
    #     return resources

    def predict(self,
                model_input: list[Resource],
                params: dict[str, Any] | None = None) -> PredictionResult:
        """
        Main prediction entry point.

        Routes to appropriate prediction method based on params['mode'].
        DO NOT override this method - implement specific predict_* methods instead.

        Args:
            model_input: List of Resource objects to process
            params: Optional configuration dictionary with keys:
                   - mode (str): Prediction mode (default: 'standard')
                   - confidence_threshold (float): Filter by confidence
                   - batch_size (int): Batch size for processing
                   - render_annotation (bool): Return rendered images
                   - device (str): Computation device
                   + mode-specific parameters (see individual method docs)

        Returns:
            List of annotation lists (one per resource), or rendered outputs
            if render_annotation=True

        Raises:
            ValueError: If mode is invalid or required parameters are missing
            NotImplementedError: If requested mode is not implemented
        """
        params = params or {}
        # model_input = self._preprocess_input(model_input, params)

        # Parse and validate mode
        mode = self._parse_mode(model_input=model_input, params=params)

        # Route to appropriate prediction method
        try:
            if not self._is_mode_implemented(mode):
                if self._is_mode_implemented(PredictionMode.DEFAULT):
                    logger.info(f"Mode '{mode.value}' not implemented, falling back to default")
                    mode = PredictionMode.DEFAULT
                else:
                    raise NotImplementedError
            logger.debug(f"Routing to '{mode.value}' mode for {len(model_input)} resources")
            result = self._route_prediction(model_input, mode, params)

            # Apply common post-processing
            result = self._post_process(result, model_input, params)

            return result

        except NotImplementedError:
            available = self.get_supported_modes()
            raise NotImplementedError(
                f"Prediction mode '{mode.value}' is not supported by this model.\n"
                f"Supported modes: {', '.join(available)}\n"
                f"Implement predict_{mode.value}() to add support for this mode."
            )

    def _parse_mode(self,
                    params: dict[str, Any],
                    model_input: list[Resource] | None = None) -> PredictionMode:
        """Parse and validate prediction mode from params."""
        mode_str = params.get('mode', PredictionMode.DEFAULT.value)
        try:
            is_all_image = all(res.mimetype.startswith('image/') for res in model_input) if model_input else False
        except Exception:
            is_all_image = False

        logger.debug(f"Parsing prediction mode: '{mode_str}' | {is_all_image=}")

        if mode_str == PredictionMode.DEFAULT.value and is_all_image:
            mode_str = PredictionMode.IMAGE.value

        try:
            return PredictionMode(mode_str)
        except ValueError:
            valid_modes = [m.value for m in PredictionMode]
            raise ValueError(
                f"Invalid prediction mode: '{mode_str}'\n"
                f"Valid modes: {', '.join(valid_modes)}"
            )

    def _route_prediction(self,
                          model_input: list[Resource],
                          mode: PredictionMode,
                          params: dict[str, Any]) -> PredictionResult:
        """Route to the appropriate prediction method based on mode."""

        # Extract mode-specific parameters and remove from kwargs
        mode_params, common_params = self._extract_mode_params(mode, params)

        # Dispatch to appropriate method
        method = self._get_method_for_mode(mode)

        # if method is None or not self._is_mode_implemented(mode, method):
        #     raise NotImplementedError

        # Call with explicit parameters
        return method(model_input, **mode_params, **common_params)

    def _extract_mode_params(self, mode: PredictionMode, params: dict[str, Any]) -> tuple[dict, dict]:
        """
        Extract mode-specific and common parameters.

        Returns:
            Tuple of (mode_specific_params, common_params)
        """
        # Define mode-specific parameter mappings
        mode_param_keys = {
            PredictionMode.FRAME: ['frame_index'],
            PredictionMode.FRAME_RANGE: ['start_frame', 'end_frame', 'step'],
            PredictionMode.SLICE: ['slice_index', 'axis'],
            PredictionMode.SLICE_RANGE: ['start_index', 'end_index', 'axis', 'step'],
            PredictionMode.PRIMARY_SLICE: ['axis'],
            PredictionMode.INTERACTIVE: ['prompt'],
            PredictionMode.FEW_SHOT: ['context_resources', 'k'],
            PredictionMode.TEMPORAL_SEQUENCE: ['center_frame', 'window_size'],
            PredictionMode.IMAGE: [],
        }

        reserved_keys = {'mode', 'confidence_threshold'}

        # Extract parameters
        mode_specific = {}
        common = {}

        mode_keys = set(mode_param_keys.get(mode, ()))

        for key, value in params.items():
            if key in reserved_keys:
                continue  # Skip mode itself and post-processing-only params
            if key in mode_keys:
                mode_specific[key] = value
            else:
                common[key] = value

        return mode_specific, common

    def _get_method_for_mode(self, mode: PredictionMode):
        """Get the method corresponding to the given prediction mode."""
        method_name = f"predict_{mode.value}"
        method = getattr(self, method_name, None)
        return method

    def get_supported_modes(self) -> list[str]:
        """
        Get list of prediction modes supported by this model.

        Returns:
            List of mode names (strings)
        """
        if self._supported_modes_cache is not None:
            return self._supported_modes_cache

        supported = []
        for mode in PredictionMode:
            if self._is_mode_implemented(mode):
                supported.append(mode.value)

        self._supported_modes_cache = supported
        return supported

    def _is_mode_implemented(self, mode: PredictionMode) -> bool:
        """Determine whether the given mode has a concrete implementation."""
        method = self._get_method_for_mode(mode)
        if method is None:
            return False

        # Check if method is from DatamintModel base class (not overridden)
        if hasattr(DatamintModel, method.__name__):
            self._get_method_for_mode
            base_method = getattr(DatamintModel, method.__name__)
            # Method is implemented if it's not the same as base class method
            return method.__func__ is not base_method

        return True

    def _post_process(self,
                      predictions: PredictionResult,
                      resources: list[Resource],
                      params: dict[str, Any]) -> PredictionResult:
        """Apply common post-processing based on params."""

        # Apply confidence threshold filtering
        conf_threshold = params.get('confidence_threshold')
        if conf_threshold is not None:
            predictions = [
                [ann for ann in pred_list
                 if getattr(ann, 'confiability', 1.0) >= conf_threshold]
                for pred_list in predictions
            ]
            logger.debug(f"Applied confidence threshold: {conf_threshold}")

        return predictions

    def predict_default(self,
                        model_input: list[Resource],
                        **kwargs) -> PredictionResult:
        """
        **OPTIONAL**: Default prediction on entire resources.

        This is the default mode and serves as fallback for unimplemented modes.
        Override this method to implement default prediction behavior.
        If called without being overridden, raises NotImplementedError.

        Args:
            model_input: Resources to process
            **kwargs: Additional user-defined parameters

        Returns:
            List of annotation lists, one per resource

        Example:
            ```python
            def predict_default(self, model_input, **kwargs):
                dataset = MyDataset(model_input)
                dataloader = DataLoader(dataset)
                model = self.mlflow_models['model'].get_raw_model()

                predictions = []
                for batch in dataloader:
                    outputs = model(batch)
                    predictions.extend(self._outputs_to_annotations(outputs))

                return predictions
            ```
        """
        raise NotImplementedError(
            "predict_default() must be implemented in your DatamintModel subclass. "
            "This is the default fallback mode for prediction."
        )

    # ========================================================================
    # VIDEO/TEMPORAL MODES
    # ========================================================================

    def predict_frame(self,
                      model_input: list[Resource],
                      frame_index: int,
                      **kwargs) -> PredictionResult:
        """
        Process specific frame from video resources.

        Args:
            model_input: Video resources
            frame_index: Index of frame to extract and process (0-based)

        Returns:
            Annotations for the specified frame (one list per resource)

        Example:
            ```python
            # Extract frame 42 from multiple videos
            predictions = model.predict(
                videos,
                params={'mode': 'frame', 'frame_index': 42}
            )
            ```
        """
        logger.warning(f"predict_frame not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)

    def predict_frame_range(self,
                            model_input: list[Resource],
                            start_frame: int,
                            end_frame: int,
                            step: int = 1,
                            **kwargs) -> PredictionResult:
        """
        Process range of frames from video resources.

        Args:
            model_input: Video resources
            start_frame: Start frame index (inclusive)
            end_frame: End frame index (inclusive)
            step: Step size between frames (default: 1)

        Returns:
            Annotations for frames in range (may be frame-scoped annotations)

        Example:
            ```python
            # Process frames 0-100, every 10th frame
            predictions = model.predict(
                videos,
                params={'mode': 'frame_range', 
                       'start_frame': 0, 
                       'end_frame': 100,
                       'step': 10}
            )
            ```
        """
        logger.warning(f"predict_frame_range not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)

    def predict_frame_interval(self,
                               model_input: list[Resource],
                               interval: int,
                               start_frame: int = 0,
                               end_frame: int | None = None,
                               **kwargs) -> PredictionResult:
        """
        Process every nth frame from video resources.

        Args:
            model_input: Video resources
            interval: Process every nth frame
            start_frame: Starting frame index
            end_frame: Ending frame index (None = last frame)

        Returns:
            Annotations for sampled frames

        Example:
            ```python
            # Process every 30th frame (1 fps for 30fps video)
            predictions = model.predict(
                videos,
                params={'mode': 'frame_interval', 'interval': 30}
            )
            ```
        """
        logger.warning(f"predict_frame_interval not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)

    def predict_all_frames(self,
                           model_input: list[Resource],
                           **kwargs) -> PredictionResult:
        """
        Process all frames independently.

        Args:
            model_input: Video resources

        Returns:
            Annotations for all frames (likely frame-scoped)

        Example:
            ```python
            # Analyze every frame
            predictions = model.predict(
                videos,
                params={'mode': 'all_frames'}
            )
            ```
        """
        logger.warning(f"predict_all_frames not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)

    # ========================================================================
    # 3D VOLUME MODES
    # ========================================================================

    def predict_slice(self,
                      model_input: list[Resource],
                      slice_index: int,
                      axis: str = 'axial',
                      **kwargs) -> PredictionResult:
        """
        Process specific slice from 3D volume.

        Args:
            model_input: 3D volume resources (DICOM series, NIfTI, etc.)
            slice_index: Index of slice to extract
            axis: Anatomical axis ('axial', 'sagittal', 'coronal')

        Returns:
            Annotations for the specified slice

        Example:
            ```python
            # Extract and analyze axial slice 50
            predictions = model.predict(
                ct_scans,
                params={'mode': 'slice', 
                       'slice_index': 50, 
                       'axis': 'axial'}
            )
            ```
        """
        logger.warning(f"predict_slice not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)

    def predict_slice_range(self,
                            model_input: list[Resource],
                            start_index: int,
                            end_index: int,
                            axis: str = 'axial',
                            step: int = 1,
                            **kwargs) -> PredictionResult:
        """
        Process range of slices from 3D volume.

        Args:
            model_input: 3D volume resources
            start_index: Start slice index (inclusive)
            end_index: End slice index (inclusive)
            axis: Anatomical axis
            step: Step size between slices

        Returns:
            Annotations for slices in range
        """
        logger.warning(f"predict_slice_range not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)

    def predict_volume(self,
                       model_input: list[Resource],
                       **kwargs) -> PredictionResult:
        """
        Process entire 3D volume.

        For true 3D models (not slice-by-slice).

        Args:
            model_input: 3D volume resources

        Returns:
            3D annotations for entire volume
        """
        logger.warning(f"predict_volume not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)

    # ========================================================================
    # ADVANCED MODES
    # ========================================================================

    def predict_interactive(self,
                            model_input: list[Resource],
                            prompt: dict[str, Any],
                            **kwargs) -> PredictionResult:
        """
        Interactive prediction with user prompts.

        For models like Segment Anything (SAM) that accept user guidance.

        Args:
            model_input: Resources to process
            prompt: Prompt dictionary with keys:
                   - 'points': list of [x, y] coordinates
                   - 'labels': list of labels (1=foreground, 0=background)
                   - 'boxes': list of [x1, y1, x2, y2] bounding boxes
                   - 'masks': list of binary mask arrays

        Returns:
            Annotations based on prompts

        Example:
            ```python
            # Segment based on positive and negative points
            predictions = model.predict(
                images,
                params={'mode': 'interactive',
                       'prompt': {
                           'points': [[100, 150], [200, 250]],
                           'labels': [1, 0]  # foreground, background
                       }}
            )
            ```
        """
        logger.warning(f"predict_interactive not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)

    def predict_few_shot(self,
                         model_input: list[Resource],
                         context_resources: list[Resource],
                         k: int = 5,
                         **kwargs) -> PredictionResult:
        """
        Few-shot prediction with context examples.

        For models that can adapt based on a few labeled examples.

        Args:
            model_input: Resources to annotate
            context_resources: Resources with existing annotations to use as examples
            k: Number of examples to use (if more are provided)

        Returns:
            Annotations informed by context examples

        Example:
            ```python
            # Predict using similar annotated examples
            predictions = model.predict(
                new_images,
                params={'mode': 'few_shot',
                       'context_resources': annotated_examples,
                       'k': 3}
            )
            ```
        """
        logger.warning(f"predict_few_shot not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)

    def predict_image(self,
                      model_input: list[Resource],
                      **kwargs) -> PredictionResult:
        """
        Process single 2D image resources.

        Args:
            model_input: 2D image resources

        Returns:
            Annotations for each image
        """
        logger.warning(f"predict_image not implemented, falling back to predict_default")
        return self.predict_default(model_input, **kwargs)
