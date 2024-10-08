import logging
from .exp_api_handler import ExperimentAPIHandler
from datetime import datetime
from typing import List, Dict, Optional, Union
from pytorch_lightning.loggers import WandbLogger, CometLogger
from collections import defaultdict
import torch
from io import BytesIO


_LOGGER = logging.getLogger(__name__)


class Experiment:
    def __init__(self,
                 name: str,
                 dataset_id: str,
                 description: Optional[str] = None,
                 api_key: Optional[str] = None,
                 root_url: Optional[str] = None) -> None:
        self.name = name
        self.apihandler = ExperimentAPIHandler(api_key=api_key, root_url=root_url)
        self.cur_step = None
        self.cur_epoch = None
        self.summary_log = defaultdict(dict)
        # self.loghandler = LogRequestHandler()

        Experiment._set_singleton_experiment(self)

        self.exp_id = self.apihandler.create_experiment(dataset_id=dataset_id,
                                                        name=name,
                                                        description=description,
                                                        environment={})  # TODO: Add environment

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
                self.summary_log['metrics'][name] = value

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

    def log_summary(self,
                    result_summary: Dict) -> None:
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

    def finish(self):
        _LOGGER.info("Finishing experiment")
        # self.apihandler.finish_experiment(self.exp_id)
        self.log_summary(result_summary=self.summary_log)


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
