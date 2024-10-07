from datamintapi import APIHandler
from typing import Optional, Dict, List, Union
import json
import logging

_LOGGER = logging.getLogger(__name__)


class ExperimentAPIHandler(APIHandler):
    def __init__(self,
                 root_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        super().__init__(root_url=root_url, api_key=api_key)
        self.exp_url = f"{self.root_url}/experiments"

    def create_experiment(self,
                          dataset_id: str,
                          name: str,
                          description: str,
                          environment: Dict) -> str:
        request_params = {
            'method': 'POST',
            'url': self.exp_url,
            'json': {"dataset_id": dataset_id,
                     "name": name,
                     "description": description,
                     "environment": environment
                     }
        }

        _LOGGER.debug(f"Creating experiment with name {name} and params {json.dumps(request_params)}")

        response = self._run_request(request_params)

        return response.json()['id']

    def get_experiment_by_id(self, exp_id: str) -> Dict:
        request_params = {
            'method': 'GET',
            'url': f"{self.exp_url}/{exp_id}"
        }

        response = self._run_request(request_params)

        return response.json()

    def get_experiments(self) -> List[Dict]:
        request_params = {
            'method': 'GET',
            'url': self.exp_url
        }

        response = self._run_request(request_params)

        return response.json()

    def log_summary(self,
                    exp_id: str,
                    result_summary: Dict,
                    ) -> None:
        request_params = {
            'method': 'POST',
            'url': f"{self.exp_url}/{exp_id}/summary",
            'json': {"result_summary": result_summary}
        }

        resp = self._run_request(request_params)

    def update_experiment(self,
                          exp_id: str,
                          name: Optional[str] = None,
                          description: Optional[str] = None,
                          result_summary: Optional[Dict] = None) -> None:

        # check that at least one of the optional parameters is not None
        if not any([name, description, result_summary]):
            return

        data = {}

        if name is not None:
            data['name'] = name
        if description is not None:
            data['description'] = description
        if result_summary is not None:
            data['result_summary'] = result_summary

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }

        request_params = {
            'method': 'PATCH',
            'url': f"{self.exp_url}/{exp_id}",
            'json': data,
            'headers': headers
        }

        resp = self._run_request(request_params)

    def log_entry(self,
                  exp_id: str,
                  entry: Dict):

        if not isinstance(entry, dict):
            raise ValueError(f"Invalid type for entry: {type(entry)}")

        request_params = {
            'method': 'POST',
            'url': f"{self.exp_url}/{exp_id}/log",
            'json': entry
        }

        _LOGGER.debug(f'logging entry with params: {json.dumps(request_params)}')

        resp = self._run_request(request_params)
        return resp

    def finish_experiment(self, exp_id: str):
        _LOGGER.info(f"Finishing experiment with id {exp_id}")
        _LOGGER.warning("Finishing experiment not implemented yet")
        # request_params = {
        #     'method': 'POST',
        #     'url': f"{self.exp_url}/{exp_id}/finish"
        # }

        # resp = self._run_request(request_params)
