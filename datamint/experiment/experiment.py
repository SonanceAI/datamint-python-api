import logging
from datamint.apihandler.api_handler import APIHandler
from datamint.apihandler.base_api_handler import DatamintException
from datetime import datetime, timezone
from typing import List, Dict, Optional, Union, Any, Tuple, IO, Literal
from collections import defaultdict
from datamint.dataset.dataset import DatamintDataset
import os
import numpy as np
import heapq
from datamint.utils import io_utils

_LOGGER = logging.getLogger(__name__)


IMPORTANT_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1score', 'Positive Predictive Value', 'Sensitivity']
IMPORTANT_METRICS = ['test/'+m.lower() for m in IMPORTANT_METRICS]
METRIC_RENAMER = {
    'precision': 'Positive Predictive Value',
    'recall': 'Sensitivity',
}


class TopN:
    class _Item:
        def __init__(self, key, item):
            self.key = key
            self.item = item

        def __lt__(self, other):
            return self.key < other.key

        def __eq__(self, other):
            return self.key == other

    def __init__(self, N, key=lambda x: x, reverse=False):
        self.N = N
        self.reverse = reverse
        self.key = key
        self.heap = []

    def add(self, item):
        item_key = float(self.key(item))
        if self.reverse:
            item_key = -item_key  # Invert the key to keep the lowest ones
        if len(self.heap) < self.N:
            heapq.heappush(self.heap, TopN._Item(item_key, item))
        else:
            heapq.heappushpop(self.heap, TopN._Item(item_key, item))

    def __len__(self):
        return len(self.heap)

    def get_top(self) -> list:
        sorted_items = sorted(self.heap, key=lambda x: x.key, reverse=True)
        return [item.item for item in sorted_items]


class _DryRunExperimentAPIHandler(APIHandler):
    """
    Dry-run implementation of the ExperimentAPIHandler.
    No data will be uploaded to the platform.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, check_connection=False, **kwargs)

    def create_experiment(self, dataset_id: str, name: str, description: str, environment: Dict) -> str:
        return "dry_run"

    def log_entry(self, exp_id: str, entry: Dict):
        pass

    def log_summary(self, exp_id: str, result_summary: Dict):
        pass

    def log_model(self, exp_id: str, *args, **kwargs):
        return {'id': 'dry_run'}

    def finish_experiment(self, exp_id: str):
        pass

    def upload_segmentations(self, *args, **kwargs) -> str:
        return "dry_run"


def _get_confidence_callback(pred) -> float:
    return pred['predicted'][0]['confidence']


class Experiment:
    """
    Experiment class to log metrics, models, and other information to the platform.

    Args:
        name (str): Name of the experiment.
        project_name (str): Name of the project.
        description (str): Description of the experiment.
        api_key (str): API key for the platform.
        root_url (str): Root URL of the platform.
        dataset_dir (str): Directory to store the datasets.
        log_enviroment (bool): Log the enviroment information.
        dry_run (bool): Run in dry-run mode. No data will be uploaded to the platform
        auto_log (bool): Automatically log the experiment using patching mechanisms.
        tags (List[str]): Tags to add to the experiment.
    """

    DATAMINT_DEFAULT_DIR = ".datamint"
    DATAMINT_DATASETS_DIR = 'datasets'

    def __init__(self,
                 name: str,
                 project_name: Optional[str] = None,
                 description: Optional[str] = None,
                 api_key: Optional[str] = None,
                 root_url: Optional[str] = None,
                 dataset_dir: Optional[str] = None,
                 log_enviroment: bool = True,
                 dry_run: bool = False,
                 auto_log=True,
                 tags: Optional[List[str]] = None,
                 allow_existing: bool = False
                 ) -> None:
        import torch
        from ._patcher import initialize_automatic_logging
        if auto_log:
            initialize_automatic_logging()
        self.auto_log = auto_log
        self.name = name
        self.dry_run = dry_run
        if dry_run:
            self.apihandler = _DryRunExperimentAPIHandler(api_key=api_key, root_url=root_url)
            _LOGGER.warning("Running in dry-run mode. No data will be uploaded to the platform.")
        else:
            self.apihandler = APIHandler(api_key=api_key, root_url=root_url)
        self.cur_step = None
        self.cur_epoch = None
        self.summary_log = defaultdict(dict)
        self.finish_callbacks = []
        self.model: torch.nn.Module = None
        self.model_id = None
        self.model_hyper_params = None
        self.is_finished = False
        self.log_enviroment = log_enviroment

        if dataset_dir is None:
            # store them in the home directory
            dataset_dir = os.path.join(os.path.expanduser("~"),
                                       Experiment.DATAMINT_DEFAULT_DIR)
            dataset_dir = os.path.join(dataset_dir, Experiment.DATAMINT_DATASETS_DIR)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        self.dataset_dir = dataset_dir

        self.project = self.apihandler.get_project_by_name(project_name)
        if 'error' in self.project:
            raise DatamintException(str(self.project))
        exp_info = self.apihandler.get_experiment_by_name(name, self.project)

        self.project_name = self.project['name']
        dataset_info = self.apihandler.get_dataset_by_id(self.project['dataset_id'])
        self.dataset_id = dataset_info['id']
        self.dataset_info = dataset_info

        if exp_info is None:
            self._initialize_new_exp(project_name, name, description, tags, log_enviroment)
        else:
            if not allow_existing:
                raise DatamintException(f"Experiment with name '{name}' already exists for project '{project_name}'.")
            self._init_from_existing_experiment(project=self.project, exp=exp_info)

        self.time_finished = None

        self.highest_predictions = defaultdict(lambda: TopN(5, key=_get_confidence_callback, reverse=False))
        self.lowest_predictions = defaultdict(lambda: TopN(5, key=_get_confidence_callback, reverse=True))

        Experiment._set_singleton_experiment(self)

    def _initialize_new_exp(self,
                            project: Dict,
                            name: str,
                            description: str,
                            tags: Optional[List[str]] = None,
                            log_enviroment: bool = True):
        env_info = Experiment.get_enviroment_info() if log_enviroment else {}
        self.exp_id = self.apihandler.create_experiment(dataset_id=self.dataset_id,
                                                        name=name,
                                                        description=description,
                                                        environment=env_info)
        self.time_started = datetime.now(timezone.utc)  # FIXME: use created_at field from response
        if tags is not None:
            self.apihandler.log_entry(exp_id=self.exp_id,
                                      entry={'tags': list(tags)})

    def _init_from_existing_experiment(self, project: Dict, exp: Dict):
        self.exp_id = exp['id']

        # raise error if the experiment is already finished
        if exp['completed_at'] is not None:
            project_name = project["name"]
            raise DatamintException(f"Experiment '{self.name}' from project '{project_name}' is already finished.")

        # example of `exp['created_at']`: 2024-11-01T19:26:12.239Z
        # example 2: 2024-11-14T17:47:22.363452-03:00
        self.time_started = datetime.fromisoformat(exp['created_at'].replace('Z', '+00:00'))

    @staticmethod
    def get_enviroment_info() -> Dict[str, Any]:
        """
        Get the enviroment information of the machine such as OS, Python version, etc.

        Returns:
            Dict: Enviroment information.
        """
        import platform
        import torchvision
        import psutil
        import socket
        import torch

        # find all ip address, removing localhost
        ip_addresses = [addr.address for iface in psutil.net_if_addrs().values()
                        for addr in iface if addr.family == socket.AF_INET and not addr.address.startswith('127.0.')]
        ip_addresses = list(set(ip_addresses))
        if len(ip_addresses) == 1:
            ip_addresses = ip_addresses[0]

        # Get the enviroment and machine information, such as OS, Python version, machine name, RAM size, etc.
        env = {
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'torchvision_version': torchvision.__version__,
            'numpy_version': np.__version__,
            'os': platform.system(),
            'os_version': platform.version(),
            'os_name': platform.system(),
            'machine_name': platform.node(),
            'cpu': platform.processor(),
            'ram_gb': psutil.virtual_memory().total / (1024. ** 3),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count(),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / (1024. ** 3) if torch.cuda.is_available() else None,
            'processor_count': os.cpu_count(),
            'processor_name': platform.processor(),
            'hostname': os.uname().nodename,
            'ip_address': ip_addresses,
        }

        return env

    def set_model(self,
                  model,
                  hyper_params: Optional[Dict] = None):
        """
        Set the model and hyper-parameters of the experiment.

        Args:
            model (torch.nn.Module): The model to log.
            hyper_params (Optional[Dict]): The hyper-parameters of the model.
        """
        self.model = model
        self.model_hyper_params = hyper_params

    @staticmethod
    def _get_dataset_info(apihandler: APIHandler,
                          dataset_id,
                          project_name: str) -> Dict:
        if project_name is not None:
            project = apihandler.get_project_by_name(project_name)
            if 'error' in project:
                raise ValueError(str(project))
            dataset_id = project['dataset_id']

        if dataset_id is None:
            raise ValueError("Either project_name or dataset_id must be provided.")

        return apihandler.get_dataset_by_id(dataset_id)

    @staticmethod
    def get_singleton_experiment() -> 'Experiment':
        global _EXPERIMENT
        return _EXPERIMENT

    @staticmethod
    def _set_singleton_experiment(experiment: 'Experiment'):
        global _EXPERIMENT
        if _EXPERIMENT is not None:
            _LOGGER.warning(
                "There is already an active Experiment. Setting a new Experiment will overwrite the existing one."
            )

        _EXPERIMENT = experiment

    def _set_step(self, step: Optional[int]) -> int:
        """
        Set the current step of the experiment and return it.
        If step is None, return the current step.
        """
        if step is not None:
            self.cur_step = step
        return self.cur_step

    def _set_epoch(self, epoch: Optional[int]) -> int:
        """
        Set the current epoch of the experiment and return it.
        If epoch is None, return the current epoch.
        """
        if epoch is not None:
            _LOGGER.debug(f"Setting current epoch to {epoch}")
            self.cur_epoch = epoch
        return self.cur_epoch

    def log_metric(self,
                   name: str,
                   value: float,
                   step: int = None,
                   epoch: int = None,
                   show_in_summary: bool = False) -> None:
        """
        Log a metric to the platform.

        Args:
            name (str): Arbritary name of the metric.
            value (float): Value of the metric.
            step (int): The step of the experiment.
            epoch (int): The epoch of the experiment.
            show_in_summary (bool): Show the metric in the summary. Use this to show only important metrics in the summary.

        Example:
            >>> exp.log_metric('test/sensitivity', 0.9, show_in_summary=True)

        See Also:
            :py:meth:`~log_metrics`
        """
        self.log_metrics({name: value},
                         step=step,
                         epoch=epoch,
                         show_in_summary=show_in_summary)

    def log_metrics(self,
                    metrics: Dict[str, float],
                    step=None,
                    epoch=None,
                    show_in_summary: bool = False) -> None:
        """
        Log multiple metrics to the platform. Handy for logging multiple metrics at once.

        Args:
            metrics (Dict[str, float]): A dictionary of metrics to log.
            step (int): The step of the experiment.
            epoch (int): The epoch of the experiment.
            show_in_summary (bool): Show the metric in the summary. Use this to show only important metrics in the summary

        Example:
            >>> exp.log_metrics({'test/loss': 0.1, 'test/accuracy': 0.9}, show_in_summary=True)

        See Also:
            :py:meth:`~log_metric`
        """
        step = self._set_step(step)
        epoch = self._set_epoch(epoch)

        # Fix nan values
        for name, value in metrics.items():
            if np.isnan(value):
                _LOGGER.debug(f"Metric {name} has a nan value. Replacing with 'NAN'.")
                metrics[name] = 'NAN'

        for name, value in metrics.items():
            spl_name = name.lower().split('test/', maxsplit=1)
            if spl_name[-1] in METRIC_RENAMER:
                name = spl_name[0] + 'test/' + METRIC_RENAMER[spl_name[-1]]

            if show_in_summary or name.lower() in IMPORTANT_METRICS:
                self.add_to_summary({'metrics': {name: value}})

        entry = [{'type': 'metric',
                  'name': name,
                  'value': value}
                 for name, value in metrics.items()]

        for m in entry:
            if step is not None:
                m['step'] = step
            if epoch is not None:
                m['epoch'] = epoch

        self.apihandler.log_entry(exp_id=self.exp_id,
                                  entry={'logs': entry})

    def add_to_summary(self,
                       dic: Dict):
        for key, value in dic.items():
            if key not in self.summary_log:
                self.summary_log[key] = value
                continue
            cur_value = self.summary_log[key]
            if isinstance(value, dict) and isinstance(cur_value, dict):
                self.summary_log[key].update(value)
            elif isinstance(value, list) and isinstance(cur_value, list):
                self.summary_log[key].extend(value)
            elif isinstance(value, tuple) and isinstance(cur_value, tuple):
                self.summary_log[key] += value
            else:
                _LOGGER.warning(f"Key {key} already exists in summary. Overwriting value.")
                self.summary_log[key] = value

    def update_summary_metrics(self,
                               phase: str | None,
                               f1score: float | None,
                               accuracy: float | None,
                               sensitivity: float | None,
                               ppv: float | None,
                               ):
        """
        Handy method to update the summary with the most common classification metrics.

        Args:
            phase (str): The phase of the experiment. Can be 'train', 'val', 'test', '', or None.
            f1score (float): The F1 score.
            accuracy (float): The accuracy.
            sensitivity (float): The sensitivity (a.k.a recall).
            specificity (float): The specificity.
            ppv (float): The positive predictive value (a.k.a precision).
        """

        if phase is None:
            phase = ""

        if phase not in ['train', 'val', 'test', '']:
            raise ValueError(f"Invalid phase: '{phase}'. Must be one of ['train', 'val', 'test', '']")

        metrics = {}
        if f1score is not None:
            metrics[f'{phase}/F1Score'] = f1score
        if accuracy is not None:
            metrics[f'{phase}/Accuracy'] = accuracy
        if sensitivity is not None:
            metrics[f'{phase}/Sensitivity'] = sensitivity
        if ppv is not None:
            metrics[f'{phase}/Positive Predictive Value'] = ppv

        self.add_to_summary({'metrics': metrics})

    def log_summary(self,
                    result_summary: Dict) -> None:
        """
        Log the summary of the experiment. This is what will be shown in the platform summary.

        Args:
            result_summary (Dict): The summary of the experiment.

        Example:
            .. code-block:: python

                exp.log_summary({"metrics": {
                                    "test/F1Score": 0.85,
                                    "test/Accuracy": 0.9,
                                    "test/Sensitivity": 0.92,
                                    "test/Positive Predictive Value": 0.79,
                                    }
                                })
        """
        _LOGGER.debug(f"Logging summary: {result_summary}")
        self.apihandler.log_summary(exp_id=self.exp_id,
                                    result_summary=result_summary)

    def log_model(self,
                  model: Any | str | IO[bytes],
                  hyper_params: Optional[Dict] = None,
                  log_model_attributes: bool = True,
                  torch_save_kwargs: Dict = {}):
        """
        Log the model to the platform.

        Args:
            model (torch.nn.Module | str | IO[bytes]): The model to log. Can be a torch model, a path to a .pt or .pth file, or a BytesIO object.
            hyper_params (Optional[Dict]): The hyper-parameters of the model. Arbitrary key-value pairs.
            log_model_attributes (bool): Adds the attributes of the model to the hyper-parameters.
            torch_save_kwargs (Dict): Additional arguments to pass to `torch.save`.

        Example:
            .. code-block:: python

                    exp.log_model(model, hyper_params={"num_layers": 3, "pretrained": True})

        """
        import torch
        if self.model_id is not None:
            raise Exception("Model is already logged. Updating the model is not supported.")

        if self.model is None:
            self.model = model
            self.model_hyper_params = hyper_params

        if log_model_attributes and isinstance(model, torch.nn.Module):
            if hyper_params is None:
                hyper_params = {}
            hyper_params['__model_classname'] = model.__class__.__name__
            # get all attributes of the model that are int, float or string
            for attr_name, attr_value in model.__dict__.items():
                if attr_name.startswith('_'):
                    continue
                if attr_name in ['training']:
                    continue
                if isinstance(attr_value, (int, float, str)):
                    hyper_params[attr_name] = attr_value

        # Add additional useful information
        if isinstance(model, torch.nn.Module):
            hyper_params.update({
                '__num_layers': len(list(model.children())),
                '__num_parameters': sum(p.numel() for p in model.parameters()),
            })

        self.model_id = self.apihandler.log_model(exp_id=self.exp_id,
                                                  model=model,
                                                  hyper_params=hyper_params,
                                                  torch_save_kwargs=torch_save_kwargs)['id']

    def _add_finish_callback(self, callback):
        self.finish_callbacks.append(callback)

    def log_dataset_stats(self, dataset: DatamintDataset,
                          dataset_entry_name: str = 'default'):
        """
        Log the statistics of the dataset.

        Args:
            dataset (DatamintDataset): The dataset to log the statistics.
            dataset_entry_name (str): The name of the dataset entry.
                Used to distinguish between different datasets and dataset splits.

        Example:
            .. code-block:: python

                dataset = exp.get_dataset(split='train')
                exp.log_dataset_stats(dataset, dataset_entry_name='train')
        """

        if dataset_entry_name is None:
            dataset_entry_name = 'default'

        dataset_stats = {
            'num_samples': len(dataset),
            'num_frame_labels': len(dataset.frame_labels_set),
            'num_segmentation_labels': len(dataset.segmentation_labels_set),
            'frame_label_distribution': dataset.get_framelabel_distribution(normalize=True),
            'segmentation_label_distribution': dataset.get_segmentationlabel_distribution(normalize=True),
        }

        keys_to_get = ['updated_at', 'total_resource']
        dataset_stats.update({k: v for k, v in dataset.metainfo.items() if k in keys_to_get})

        self.add_to_summary({'dataset_stats': dataset_stats})
        dataset_params_names = ['return_dicom', 'return_metainfo', 'return_segmentations'
                                'return_frame_by_frame', 'return_as_semantic_segmentation']
        dataset_stats['dataset_params'] = {k: getattr(dataset, k) for k in dataset_params_names if hasattr(dataset, k)}
        dataset_stats['dataset_params']['image_transform'] = repr(dataset.image_transform)
        dataset_stats['dataset_params']['mask_transform'] = repr(dataset.mask_transform)

        self.apihandler.log_entry(exp_id=self.exp_id,
                                  entry={'dataset_stats': {dataset_entry_name: dataset_stats}})

    def get_dataset(self, split: str = 'all', **kwargs) -> DatamintDataset:
        """
        Get the dataset associated with the experiment's project. 
        The dataset will be downloaded to the directory specified in the constructor (`self.dataset_dir`).

        Args:
            split (str): The split of the dataset to get. Can be one of ['all', 'train', 'test', 'val'].
            **kwargs: Additional arguments to pass to the :py:class:`~datamint.dataset.dataset.DatamintDataset` class.

        Returns:
            DatamintDataset: The dataset object.
        """
        if split not in ['all', 'train', 'test', 'val']:
            raise ValueError(f"Invalid split parameter: '{split}'. Must be one of ['all', 'train', 'test', 'val']")

        params = dict(project_name=self.project_name)

        dataset = DatamintDataset(root=self.dataset_dir,
                                  api_key=self.apihandler.api_key,
                                  server_url=self.apihandler.root_url,
                                  **params,
                                  **kwargs)

        # infer task
        if not hasattr(self, 'detected_task') and self.auto_log:
            self.detected_task = self._detect_machine_learning_task(dataset)
            self.add_to_summary({'detected_task': self.detected_task})

        if split == 'all':
            self.log_dataset_stats(dataset, split)
            return dataset

        # FIXME: samples should be marked as train, test, val previously

        train_split_val = 0.8
        test_split_val = 0.1
        indices = list(range(len(dataset)))
        rs = np.random.RandomState(42)
        rs.shuffle(indices)
        train_split_idx = int(train_split_val * len(dataset))
        test_split_idx = int(np.ceil(test_split_val * len(dataset))) + train_split_idx
        train_indices = indices[:train_split_idx]
        test_indices = indices[train_split_idx:test_split_idx]
        val_indices = indices[test_split_idx:]

        if split == 'train':
            indices_to_split = train_indices
        elif split == 'test':
            indices_to_split = test_indices
        elif split == 'val':
            indices_to_split = val_indices

        dataset = dataset.subset(indices_to_split)
        self.log_dataset_stats(dataset, split)
        return dataset

    def _detect_machine_learning_task(self, dataset: DatamintDataset) -> str:
        try:
            # Detect machine learning task based on the dataset params
            if dataset.return_as_semantic_segmentation and len(dataset.segmentation_labels_set) > 0:
                return 'semantic segmentation'
            elif dataset.return_segmentations and len(dataset.segmentation_labels_set) > 0:
                return 'instance segmentation'

            num_labels = len(dataset.frame_labels_set)  # FIXME: when not frame by frame
            num_categories = len(dataset.segmentation_labels_set)
            if num_categories == 0:
                if num_labels == 1:
                    return 'binary classification'
                elif num_labels > 1:
                    return 'multilabel classification'
            elif num_categories == 1:
                if num_labels == 0:
                    return 'multiclass classification'
                return 'multi-task classification'
            else:
                return 'multi-task classification'
        except Exception as e:
            _LOGGER.warning(f"Could not detect machine learning task: {e}")

        return 'unknown'

    def _log_predictions(self,
                         predictions: List[Dict[str, Any]],
                         dataset_split: Optional[str] = None,
                         step: Optional[int] = None,
                         epoch: Optional[int] = None):
        """
        Log the predictions of the model.

        Args:
            predictions (List[Dict[str, Any]]): The predictions to log. See example below.
            step (Optional[int]): The step of the experiment.
            epoch (Optional[int]): The epoch of the experiment.


        Example:
            .. code-block:: python

                predictions = [
                            {
                                'resource_id': '123',
                # If not provided, it will be assumed predictions are for the whole resource.
                                'frame_index': 0,
                                'predicted': [
                                    {
                                        'identifier': 'has_fracture',
                                        'value': True, # Optional
                                        'confidence': 0.9,  # Optional
                                        'ground_truth': True  # Optional
                                    },
                                    {
                                        'identifier': 'tumor',
                                        'value': segmentation1,  # Optional. numpy array of shape (H, W)
                                        'confidence': 0.9  # Optional. Can be mask.max()
                                    }
                                ]
                            }]
                exp.log_predictions(predictions)
        """

        self._set_step(step)
        self._set_epoch(epoch)

        entry = {'type': 'prediction',
                 'predictions': predictions,
                 'dataset_split': dataset_split,
                 'step': step,
                 }

        if dataset_split == 'test':
            for pred in predictions:
                # if prediction is categorical
                if pred['prediction_type'] == 'category':
                    for p in pred['predicted']:
                        if 'confidence' in p:
                            self.highest_predictions[p['identifier']].add(pred)
                            self.lowest_predictions[p['identifier']].add(pred)

        self.apihandler.log_entry(exp_id=self.exp_id,
                                  entry=entry)

    def log_classification_predictions(self,
                                       predictions_conf: np.ndarray,
                                       resource_ids: List[str],
                                       label_names: List[str],
                                       dataset_split: Optional[str] = None,
                                       frame_idxs: Optional[List[int]] = None,
                                       step: Optional[int] = None,
                                       epoch: Optional[int] = None,
                                       add_info: Optional[Dict] = None
                                       ):
        """
        Log the classification predictions of the model.

        Args:
            predictions_conf (np.ndarray): The predictions of the model. Can have two shapes:

                - Shape (N, C) where N is the number of samples and C is the number of classes.
                  Does not need to sum to 1 (i.e., can be multilabel).
                - Shape (N,) where N is the number of samples.
                  In this case, `label_names` should have the same length as the predictions.

            label_names (List[str]): The names of the classes. 
                If the predictions are shape (N,), this should have the same length as the predictions.
            resource_ids (List[str]): The resource IDs of the samples.
            dataset_split (Optional[str]): The dataset split of the predictions.
            frame_idxs (Optional[List[int]]): The frame indexes of the predictions.
            step (Optional[int]): The step of the experiment.
            epoch (Optional[int]): The epoch of the experiment.
            add_info (Optional[Dict]): Additional information to add to each prediction.

        Example:
            .. code-block:: python

                predictions_conf = np.array([[0.9, 0.1], [0.2, 0.8]])
                label_names = ['cat', 'dog']
                resource_ids = ['123', '456']
                exp.log_classification_predictions(predictions_conf, label_names, resource_ids, dataset_split='test')
        """

        # check predictions shape and lengths
        if len(predictions_conf) != len(resource_ids):
            raise ValueError("Length of predictions and resource_ids must be the same.")

        if predictions_conf.ndim == 2:
            if predictions_conf.shape[1] != len(label_names):
                raise ValueError("Number of classes must match the number of columns in predictions_conf.")
        elif predictions_conf.ndim == 1:
            if len(label_names) != len(predictions_conf):
                raise ValueError("Number of classes must match the length of predictions when predictions are 1D.")
        else:
            raise ValueError("Predictions must be 1D or 2D.")

        resources = self.apihandler.get_resources_by_ids(resource_ids)

        predictions = []
        if predictions_conf.ndim == 2:
            for res, pred in zip(resources, predictions_conf):
                data = {'resource_id': res['id'],
                        'resource_filename': res['filename'],
                        'prediction_type': 'category',
                        'predicted': [{'identifier': label_names[i],
                                       'confidence': float(pred[i])}
                                      for i in range(len(pred))]}
                if add_info is not None:
                    data.update(add_info)
                predictions.append(data)
        else:
            # if predictions are 1D, label_names have the same length
            for res, pred, label_i in zip(resources, predictions_conf, label_names):
                data = {'resource_id': res['id'],
                        'resource_filename': res['filename'],
                        'prediction_type': 'category',
                        'predicted': [{'identifier': label_i,
                                       'confidence': float(pred)}
                                      ]}
                if add_info is not None:
                    data.update(add_info)
                predictions.append(data)

        if frame_idxs is not None:
            for pred, frame_idx in zip(predictions, frame_idxs):
                pred['frame_index'] = frame_idx
        self._log_predictions(predictions, step=step, epoch=epoch, dataset_split=dataset_split)

    def log_segmentation_predictions(self,
                                     resource_id: str | dict,
                                     predictions: np.ndarray | str,
                                     label_name: str | dict[int, str],
                                     frame_index: int | list[int] | None = None,
                                     threshold: float = 0.5,
                                     predictions_format: Literal['multi-class', 'probability'] = 'probability'
                                     ):
        """
        Log the segmentation prediction of the model for a single frame

        Args:
            resource_id: The resource ID of the sample.
            predictions: The predictions of the model. One binary mask for each class. Can be a numpy array of shape (H, W) or (N,H,W);
                Or a path to a png file; Or a path to a .nii/.nii.gz file.
            label_name: The name of the class or a dictionary mapping pixel values to names.
                Example: ``{1: 'Femur', 2: 'Tibia'}`` means that pixel value 1 is 'Femur' and pixel value 2 is 'Tibia'.
            frame_index: The frame index of the prediction or a list of frame indexes.
                If a list, must have the same length as the predictions.
                If None, 
            threshold: The threshold to apply to the predictions.
            predictions_format: The format of the predictions. Can be a probability mask ('probability') or a multi-class mask ('multi-class').

        Example:
            .. code-block:: python

                resource_id = '123'
                predictions = np.array([[0.1, 0.4], [0.9, 0.2]])
                label_name = 'fracture'
                exp.log_segmentation_predictions(resource_id, predictions, label_name, threshold=0.5)

            .. code-block:: python

                resource_id = '456'
                predictions = np.array([[0, 1, 2], [1, 2, 0]])  # Multi-class mask with values 0, 1, 2
                label_name = {1: 'Femur', 2: 'Tibia'}  # Mapping of pixel values to class names
                exp.log_segmentation_predictions(
                    resource_id, 
                    predictions, 
                    label_name, 
                    predictions_format='multi-class'
                )
        """

        if predictions_format not in ['multi-class', 'probability']:
            raise ValueError("predictions_format must be 'multi-class' or 'probability'.")

        if isinstance(label_name, dict) and predictions_format!='multi-class':
            raise ValueError("If label_name is a dictionary, predictions_format must be 'multi-class'.")

        if isinstance(resource_id, dict):
            resource_id = resource_id['id']

        if self.model_id is None:
            raise ValueError("Model is not logged. Cannot log segmentation predictions. see `log_model` method.")

        if isinstance(predictions, str):
            predictions = io_utils.read_array_normalized(predictions)

        if predictions_format == 'probability':
            predictions = predictions > threshold

        is_2d_prediction = predictions.ndim == 2

        if predictions.ndim == 4 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]
        elif predictions.ndim == 2:
            predictions = predictions[np.newaxis]
        elif predictions.ndim != 3:
            raise ValueError(f"Prediction with shape {predictions.shape} is different than (H, W) and (N,H,W).")

        if frame_index is None:
            if is_2d_prediction:
                raise ValueError("frame_index must be provided when predictions is 2D.")
            frame_index = list(range(predictions.shape[0]))
        elif isinstance(frame_index, int):
            frame_index = [frame_index]
        else:
            if len(frame_index) != predictions.shape[0]:
                raise ValueError("Length of frame_index must match the first dimension of predictions.")

        new_ann_id = self.apihandler.upload_segmentations(
            resource_id=resource_id,
            file_path=predictions.transpose(1, 2, 0),
            name=label_name,
            frame_index=frame_index,
            model_id=self.model_id,
            worklist_id=self.project['worklist_id'],
        )

    def log_semantic_seg_predictions(self,
                                     predictions: np.ndarray | str,
                                     resource_ids: Union[list[str], str],
                                     label_names: list[str],
                                     dataset_split: Optional[str] = None,
                                     frame_idxs: Optional[list[int]] = None,
                                     step: Optional[int] = None,
                                     epoch: Optional[int] = None,
                                     threshold: float = 0.5
                                     ):
        """
        Log the semantic segmentation predictions of the model.

        Args:
            predictions (np.ndarray | str): The predictions of the model. A list of numpy arrays of shape (N, C, H, W).
                Or a path to a png file; Or a path to a .nii.gz file.
            label_names (list[str]): The names of the classes. List of strings of size C.
            resource_ids (list[str]): The resource IDs of the samples.
            dataset_split (Optional[str]): The dataset split of the predictions.
            frame_idxs (Optional[list[int]]): The frame indexes of the predictions.
            step (Optional[int]): The step of the experiment.
            epoch (Optional[int]): The epoch of the experiment.
        """

        if isinstance(predictions, str):
            predictions = io_utils.read_array_normalized(predictions)

        if isinstance(resource_ids, str):
            resource_ids = [resource_ids] * len(predictions)

        if predictions.ndim != 4:
            raise ValueError("Predictions must be of shape (N, C, H, W).")

        # check lengths
        if len(predictions) != len(resource_ids):
            raise ValueError("Length of predictions and resource_ids must be the same.")

        if frame_idxs is not None:
            if len(predictions) != len(frame_idxs):
                raise ValueError("Length of predictions and frame_idxs must be the same.")
            # non negative frame indexes
            if any(fidx < 0 for fidx in frame_idxs):
                raise ValueError("Frame indexes must be non-negative.")

        if len(label_names) != predictions.shape[1]:
            raise ValueError("Number of classes must match the number of columns in predictions.")

        predictions_conf = predictions.max(axis=(2, 3))  # final shape: (N, C)

        # log it as classification predictions
        self.log_classification_predictions(predictions_conf=predictions_conf,
                                            label_names=label_names,
                                            resource_ids=resource_ids,
                                            dataset_split=dataset_split,
                                            frame_idxs=frame_idxs,
                                            step=step,
                                            epoch=epoch,
                                            add_info={'origin': 'semantic segmentation'})

        if self.model_id is not None:
            _LOGGER.info("Uploading segmentation masks to the platform.")
            # For each frame
            predictions = predictions > threshold
            grouped_predictions = defaultdict(list)
            for fidx, res_id, pred in zip(frame_idxs, resource_ids, predictions):
                grouped_predictions[res_id].append((fidx, pred))

            for res_id, list_preds in grouped_predictions.items():
                frame_idxs = [fidx for fidx, _ in list_preds]
                preds = np.stack([pred for _, pred in list_preds])
                for i in range(len(label_names)):
                    preds_i = preds[:, i]  # get the i-th class predictions
                    # preds_i.shape: (N, H, W)
                    new_ann_id = self.apihandler.upload_segmentations(
                        resource_id=res_id,
                        file_path=preds_i,
                        name=label_names[i],
                        frame_index=frame_idxs,
                        model_id=self.model_id,
                        worklist_id=self.project['worklist_id'],
                    )
        else:
            _LOGGER.warning("Model is not logged. Skipping uploading segmentation masks.")

    def finish(self):
        """
        Finish the experiment.
        This will log the summary and finish the experiment.
        """
        def _process_toppredictions(top_predictions: Dict[str, TopN], rev: bool) -> Tuple[TopN, Dict]:
            preds_per_label = {key: values.get_top()
                               for key, values in top_predictions.items()}
            # get the highest prediction over all labels
            preds_combined = TopN(5, key=_get_confidence_callback, reverse=rev)
            for label_preds in preds_per_label.values():
                for pred in label_preds:
                    preds_combined.add(pred)

            return preds_combined, preds_per_label

        if self.is_finished:
            _LOGGER.debug("Experiment is already finished.")
            return
        _LOGGER.info("Finishing experiment")
        for callback in self.finish_callbacks:
            callback(self)
        self.time_finished = datetime.now(timezone.utc)
        time_spent_seconds = (self.time_finished - self.time_started).total_seconds()

        ### produce finishing summary ###
        # time spent
        self.add_to_summary({'time_spent_seconds': time_spent_seconds})

        # add the most interesting predictions
        if len(self.highest_predictions) > 0:
            highest_preds_combined, highest_preds_per_label = _process_toppredictions(self.highest_predictions, False)
            lowest_preds_combined, lowest_preds_per_label = _process_toppredictions(self.lowest_predictions, True)

            self.add_to_summary({'highest_predictions': {'combined': highest_preds_combined.get_top(),
                                                         'per_label': highest_preds_per_label
                                                         }
                                 }
                                )

            self.add_to_summary({'lowest_predictions': {'combined': lowest_preds_combined.get_top(),
                                                        'per_label': lowest_preds_per_label
                                                        }
                                 }
                                )

        self.log_summary(result_summary=self.summary_log)
        # if the model is not already logged, log it
        if self.model is not None and self.model_id is None:
            self.log_model(model=self.model, hyper_params=self.model_hyper_params)
        self.apihandler.finish_experiment(self.exp_id)
        self.is_finished = True

        _LOGGER.info("Experiment finished and uploaded to the platform.")


class _LogHistory:
    """
    TODO: integrate this with the Experiment class.
    """

    def __init__(self):
        self.history = []

    def append(self, dt: datetime = None, **kwargs):
        if dt is None:
            dt = datetime.now(timezone.utc)
        else:
            if dt.tzinfo is None:
                _LOGGER.warning("No timezone information provided. Assuming UTC.")
                dt = dt.replace(tzinfo=timezone.utc)

        item = {
            # datetime in GMT+0
            'timestamp': dt.timestamp(),
            **kwargs
        }
        self.history.append(item)

    def get_history(self) -> List[Dict]:
        return self.history


_EXPERIMENT: Experiment = None
