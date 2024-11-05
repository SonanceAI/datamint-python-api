import logging
from .exp_api_handler import ExperimentAPIHandler
from datamintapi.api_handler import DatamintException
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from collections import defaultdict
import torch
from io import BytesIO
from datamintapi import Dataset as DatamintDataset
import os
import numpy as np

_LOGGER = logging.getLogger(__name__)


class _DryRunExperimentAPIHandler(ExperimentAPIHandler):
    """
    Dry-run implementation of the ExperimentAPIHandler.
    No data will be uploaded to the platform.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_experiment(self, dataset_id: str, name: str, description: str, environment: Dict) -> str:
        return "dry_run"

    def log_entry(self, exp_id: str, entry: Dict):
        pass

    def log_summary(self, exp_id: str, result_summary: Dict):
        pass

    def log_model(self, exp_id: str, model: Union[torch.nn.Module, str, BytesIO], hyper_params: Optional[Dict] = None, torch_save_kwargs: Dict = {}):
        pass

    def finish_experiment(self, exp_id: str):
        pass


class Experiment:
    """
    Experiment class to log metrics, models, and other information to the platform.

    Args:
        name (str): Name of the experiment.
        project_name (str): Name of the project.
        dataset_id (str): ID of the dataset.
        dataset_name (str): Name of the dataset.
        description (str): Description of the experiment.
        api_key (str): API key for the platform.
        root_url (str): Root URL of the platform.
        dataset_dir (str): Directory to store the datasets.
        log_enviroment (bool): Log the enviroment information.
        dry_run (bool): Run in dry-run mode. No data will be uploaded to the platform

    """

    DATAMINT_DEFAULT_DIR = ".datamint"
    DATAMINT_DATASETS_DIR = 'datasets'

    def __init__(self,
                 name: str,
                 project_name: Optional[str] = None,
                 dataset_id: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 description: Optional[str] = None,
                 api_key: Optional[str] = None,
                 root_url: Optional[str] = None,
                 dataset_dir: Optional[str] = None,
                 log_enviroment: bool = True,
                 dry_run: bool = False
                 ) -> None:
        self.name = name
        self.dry_run = dry_run
        if dry_run:
            self.apihandler = _DryRunExperimentAPIHandler(api_key=api_key, root_url=root_url)
            _LOGGER.warning("Running in dry-run mode. No data will be uploaded to the platform.")
        else:
            self.apihandler = ExperimentAPIHandler(api_key=api_key, root_url=root_url)
        self.cur_step = None
        self.cur_epoch = None
        self.summary_log = defaultdict(dict)
        self.finish_callbacks = []
        self.model: torch.nn.Module = None
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

        dataset_info = Experiment._get_dataset_info(self.apihandler,
                                                    dataset_id,
                                                    dataset_name)
        self.dataset_id = dataset_info['id']
        self.dataset_name = dataset_info['name']
        self.dataset_info = dataset_info

        self.dataset = None
        # self.loghandler = LogRequestHandler()

        # Check if experiment already exists with the same name
        experiments = self.apihandler.get_experiments()
        for exp in experiments:
            if exp['name'] == name:
                raise DatamintException(f"Experiment with name '{name}' already exists.")

        Experiment._set_singleton_experiment(self)

        env_info = Experiment.get_enviroment_info() if log_enviroment else {}
        self.exp_id = self.apihandler.create_experiment(dataset_id=self.dataset_id,
                                                        name=name,
                                                        description=description,
                                                        environment=env_info)
        self.time_started = datetime.now(datetime.timezone.utc)
        self.time_finished = None

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

    def set_model(self, model, hyper_params=None):
        self.model = model
        self.model_hyper_params = hyper_params

    @staticmethod
    def _get_dataset_info(apihandler: ExperimentAPIHandler,
                          dataset_id,
                          dataset_name) -> Dict:
        if dataset_id is None:
            if dataset_name is None:
                raise ValueError("dataset_name or dataset_id must be provided.")
            datasets_infos = apihandler.get_datasetsinfo_by_name(dataset_name)
            if len(datasets_infos) == 0:
                raise DatamintException(f"No dataset found with name {dataset_name}")
            if len(datasets_infos) >= 2:
                raise DatamintException(f"Multiple datasets found with name {dataset_name}. Please provide dataset_id.")

            return datasets_infos[0]

        return apihandler.get_dataset_by_id(dataset_id)

    @staticmethod
    def get_singleton_experiment() -> 'Experiment':
        global EXPERIMENT
        return EXPERIMENT

    @staticmethod
    def _set_singleton_experiment(experiment: 'Experiment'):
        global EXPERIMENT
        if EXPERIMENT is not None:
            _LOGGER.warning(
                "There is already an active Experiment. Setting a new Experiment will overwrite the existing one."
            )

        EXPERIMENT = experiment

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
                   show_in_summary: Optional[bool] = None) -> None:
        self.log_metrics({name: value},
                         step=step,
                         epoch=epoch,
                         show_in_summary=show_in_summary)

    def log_metrics(self,
                    metrics: Dict[str, float],
                    step=None,
                    epoch=None,
                    show_in_summary: Optional[bool] = None) -> None:
        step = self._set_step(step)
        epoch = self._set_epoch(epoch)

        if show_in_summary == True:
            for name, value in metrics.items():
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

    def log_summary(self,
                    result_summary: Dict) -> None:
        _LOGGER.debug(f"Logging summary: {result_summary}")
        self.apihandler.log_summary(exp_id=self.exp_id,
                                    result_summary=result_summary)

    def log_model(self,
                  model: Union[torch.nn.Module, str, BytesIO],
                  hyper_params: Optional[Dict] = None,
                  log_model_attributes: bool = True,
                  torch_save_kwargs: Dict = {}):
        if self.model is None:
            self.model = model
            self.model_hyper_params = hyper_params

        if log_model_attributes:
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
        hyper_params.update({
            '__num_layers': len(list(model.children())),
            '__num_parameters': sum(p.numel() for p in model.parameters()),
        })

        self.apihandler.log_model(exp_id=self.exp_id,
                                  model=model,
                                  hyper_params=hyper_params,
                                  torch_save_kwargs=torch_save_kwargs)

    def _add_finish_callback(self, callback):
        self.finish_callbacks.append(callback)

    def _log_dataset_stats(self, dataset: DatamintDataset):
        dataset_stats = {
            'num_samples': len(dataset),
            'num_labels': len(dataset.num_labels),
            'num_segmentation_labels': dataset.num_segmentation_labels,
            'frame_label_distribution': dataset.get_framelabel_distribution(normalize=True),
            'segmentation_label_distribution': dataset.get_segmentationlabel_distribution(normalize=True),
        }

        keys_to_get = ['updated_at', 'total_resource']
        dataset_stats.update({k: v for k, v in dataset.metainfo.items() if k in keys_to_get})

        self.add_to_summary({'dataset_stats': dataset_stats})
        dataset_params_names = ['return_dicom', 'return_metainfo', 'return_seg_annotations'
                                'return_frame_by_frame', 'return_as_semantic_segmentation']
        dataset_stats['dataset_params'] = {k: getattr(dataset, k) for k in dataset_params_names if hasattr(dataset, k)}
        dataset_stats['dataset_params']['image_transform'] = repr(dataset.image_transform)
        dataset_stats['dataset_params']['mask_transform'] = repr(dataset.mask_transform)

        self.apihandler.log_entry(exp_id=self.exp_id,
                                  entry={'dataset_stats': dataset_stats})

    def get_dataset(self, split: str = 'all', **kwargs) -> DatamintDataset:
        if split not in ['all', 'train', 'test', 'val']:
            raise ValueError(f"Invalid split parameter: '{split}'. Must be one of ['all', 'train', 'test', 'val']")

        self.dataset = DatamintDataset(self.dataset_dir,
                                       dataset_name=self.dataset_name,
                                       api_key=self.apihandler.api_key,
                                       server_url=self.apihandler.root_url,
                                       return_metainfo=True,
                                       return_dicom=False,
                                       **kwargs)

        if split == 'all':
            return self.dataset

        # FIXME: samples should be marked as train, test, val previously

        train_split_val = 0.8
        test_split_val = 0.1
        indices = list(range(len(self.dataset)))
        rs = np.random.RandomState(42)
        rs.shuffle(indices)
        train_split_idx = int(train_split_val * len(self.dataset))
        test_split_idx = int(np.ceil(test_split_val * len(self.dataset))) + train_split_idx
        train_indices = indices[:train_split_idx]
        test_indices = indices[train_split_idx:test_split_idx]
        val_indices = indices[test_split_idx:]

        if split == 'train':
            indices_to_split = train_indices
        elif split == 'test':
            indices_to_split = test_indices
        elif split == 'val':
            indices_to_split = val_indices

        return self.dataset.subset(indices_to_split)

    def finish(self):
        if self.is_finished:
            _LOGGER.debug("Experiment is already finished.")
            return
        _LOGGER.info("Finishing experiment")
        for callback in self.finish_callbacks:
            callback(self)
        self.time_finished = datetime.now(datetime.timezone.utc)
        time_spent_seconds = (self.time_finished - self.time_started).total_seconds()
        self.add_to_summary({'time_spent_seconds': time_spent_seconds})
        # self.apihandler.finish_experiment(self.exp_id)
        self.log_summary(result_summary=self.summary_log)
        if self.model is not None:
            self.log_model(model=self.model, hyper_params=self.model_hyper_params)
        self.apihandler.finish_experiment(self.exp_id)
        self.is_finished = True


class LogHistory:
    def __init__(self):
        self.history = []

    def append(self, dt: datetime = None, **kwargs):
        if dt is None:
            dt = datetime.now(datetime.timezone.utc)
        else:
            if dt.tzinfo is None:
                _LOGGER.warning("No timezone information provided. Assuming UTC.")
                dt = dt.replace(tzinfo=datetime.timezone.utc)

        item = {
            # datetime in GMT+0
            'timestamp': dt.timestamp(),
            **kwargs
        }
        self.history.append(item)

    def get_history(self) -> List[Dict]:
        return self.history


EXPERIMENT: Experiment = None
