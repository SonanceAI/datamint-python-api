from datamint.apihandler.base_api_handler import BaseAPIHandler
from typing import Optional, Dict, List, Union, Any
import json
import logging
from io import BytesIO

_LOGGER = logging.getLogger(__name__)


class ExperimentAPIHandler(BaseAPIHandler):
    def __init__(self,
                 root_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 check_connection: bool = True,
                 **kwargs):
        super().__init__(root_url=root_url, api_key=api_key, check_connection=check_connection, **kwargs)
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
    
    def get_experiment_logs(self, exp_id: str) -> List[Dict]:
        request_params = {
            'method': 'GET',
            'url': f"{self.exp_url}/{exp_id}/log"
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

        resp = self._run_request(request_params)
        return resp

    def finish_experiment(self, exp_id: str):
        pass
        # _LOGGER.info(f"Finishing experiment with id {exp_id}")
        # _LOGGER.warning("Finishing experiment not implemented yet")
        # request_params = {
        #     'method': 'POST',
        #     'url': f"{self.exp_url}/{exp_id}/finish"
        # }

        # resp = self._run_request(request_params)

    def log_model(self,
                  exp_id: str,
                  model: Union[Any, str, BytesIO],
                  hyper_params: Optional[Dict] = None,
                  torch_save_kwargs: Dict = {}) -> Dict:
        import torch
        if isinstance(model, torch.nn.Module):
            f = BytesIO()
            torch.save(model, f, **torch_save_kwargs)
            f.seek(0)
            f.name = None
        elif isinstance(model, str):
            with open(model, 'rb') as f1:
                f = BytesIO(f1.read())
                f.name = None
        elif isinstance(model, BytesIO):
            f = model
        else:
            raise ValueError(f"Invalid type for model: {type(model)}")

        name = None
        f.name = name

        try:
            json_data = hyper_params
            json_data['model_name'] = name
            request_params = {
                'method': 'POST',
                'url': f"{self.exp_url}/{exp_id}/model",
                'data': json_data,
                'files': [(None, f)],
            }

            resp = self._run_request(request_params).json()
            return resp[0]
        finally:
            f.close()

    def get_experiment_by_name(self, name: str, project: Dict) -> Optional[Dict]:
        """
        Get the experiment by name of the project.

        Args:
            name (str): Name of the experiment.
            project (Dict): The project to search for the experiment.

        Returns:
            Optional[Dict]: The experiment if found, otherwise None.
        """
        # uses GET /projects/{project_id}/experiments

        project_id = project['id']
        request_params = {
            'method': 'GET',
            'url': f"{self.root_url}/projects/{project_id}/experiments"
        }

        response = self._run_request(request_params)
        experiments = response.json()
        for exp in experiments:
            if exp['name'] == name:
                return exp
        return None
