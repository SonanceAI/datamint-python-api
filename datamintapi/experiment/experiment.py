import logging
from .exp_api_handler import ExperimentAPIHandler
from datetime import datetime
from typing import List, Dict

_LOGGER = logging.getLogger(__name__)


class Experiment:
    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        # FIXME: Change to get URL from config
        root_url = 'https://stagingapi.datamint.io'
        self.apihandler = ExperimentAPIHandler(root_url=root_url)
        self.metrics = {}

        Experiment._set_singleton_experiment(self)

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

    def log_metric(self, name: str, value: float) -> None:
        _LOGGER.info(f"Logging metric {name} with value {value}")
        self.metrics[name] = value
        # TODO


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
