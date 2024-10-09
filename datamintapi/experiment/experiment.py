import logging
from .exp_api_handler import ExperimentAPIHandler
from datamintapi.api_handler import DatamintException
from datetime import datetime
from typing import List, Dict, Optional, Union
from pytorch_lightning.loggers import WandbLogger, CometLogger
from collections import defaultdict
import torch
from io import BytesIO
from datamintapi import Dataset as DatamintDataset
import os


_LOGGER = logging.getLogger(__name__)


class Experiment:

    DATAMINT_DEFAULT_DIR = ".datamint"
    DATAMINT_DATASETS_DIR = 'datasets'

    def __init__(self,
                 name: str,
                 dataset_id: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 description: Optional[str] = None,
                 api_key: Optional[str] = None,
                 root_url: Optional[str] = None,
                 dataset_dir: Optional[str] = None) -> None:
        self.name = name
        self.apihandler = ExperimentAPIHandler(api_key=api_key, root_url=root_url)
        self.cur_step = None
        self.cur_epoch = None
        self.summary_log = defaultdict(dict)
        self.finish_callbacks = []

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
        self.dataset = None
        # self.loghandler = LogRequestHandler()

        Experiment._set_singleton_experiment(self)

        self.exp_id = self.apihandler.create_experiment(dataset_id=self.dataset_id,
                                                        name=name,
                                                        description=description,
                                                        environment={})  # TODO: Add environment

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
                  torch_save_kwargs: Dict = {}):
        self.apihandler.log_model(exp_id=self.exp_id,
                                  model=model,
                                  hyper_params=hyper_params,
                                  torch_save_kwargs=torch_save_kwargs)

    def _add_finish_callback(self, callback):
        self.finish_callbacks.append(callback)

    def get_dataset(self, split: str = 'all') -> DatamintDataset:
        # FIXME: Implement split
        _LOGGER.warning("split parameter is not implemented yet. Returning the full dataset.")
        if self.dataset is None:
            self.dataset = DatamintDataset(self.dataset_dir,
                                           dataset_name=self.dataset_name,
                                           api_key=self.apihandler.api_key,
                                           server_url=self.apihandler.root_url,
                                           return_dicom=False)
        return self.dataset

    def finish(self):
        _LOGGER.info("Finishing experiment")
        for callback in self.finish_callbacks:
            callback(self)
        # self.apihandler.finish_experiment(self.exp_id)
        self.log_summary(result_summary=self.summary_log)
        self.apihandler.finish_experiment(self.exp_id)


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
