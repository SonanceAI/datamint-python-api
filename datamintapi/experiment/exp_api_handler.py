from datamintapi import APIHandler
from typing import Optional, Dict, List
import json


class ExperimentAPIHandler(APIHandler):
    def __init__(self, root_url: str, api_key: Optional[str] = None):
        super().__init__(root_url, api_key)
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
                    name: str,
                    description: str) -> None:
        request_params = {
            'method': 'POST',
            'url': f"{self.exp_url}/{exp_id}/summary",
            'json': {"name": name,
                     "description": description,
                     "result_summary": result_summary
                     }
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
