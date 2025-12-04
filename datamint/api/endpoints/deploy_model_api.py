import httpx
from ..entity_base_api import EntityBaseApi, ApiConfig
from datamint.entities.deployjob import DeployJob

class DeployModelApi(EntityBaseApi[DeployJob]):
    """API handler for model deployment endpoints."""

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None) -> None:
        super().__init__(config, DeployJob, 'datamint/api/v1/deploy-model', client)

    def get_by_id(self, entity_id: str) -> DeployJob:
        """Get deployment job status by ID."""
        response = self._make_request('GET', f'/{self.endpoint_base}/status/{entity_id}')
        data = response.json()
        if 'job_id' in data:
            data['id'] = data.pop('job_id')
        return self._init_entity_obj(**data)

    def start(self,
              model_name: str,
              model_version: int | None = None,
              model_alias: str | None = None,
              image_name: str | None = None,
              with_gpu: bool = False,
              convert_to_onnx: bool = False,
              input_shape: list[int] | None = None) -> DeployJob:
        """Start a new deployment job."""
        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "model_alias": model_alias,
            "image_name": image_name,
            "with_gpu": with_gpu,
            "convert_to_onnx": convert_to_onnx,
            "input_shape": input_shape
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        response = self._make_request('POST', f'/{self.endpoint_base}/start', json=payload)
        data = response.json()
        return self.get_by_id(data['job_id'])

    def cancel(self, job: str | DeployJob) -> bool:
        """Cancel a deployment job."""
        job_id = self._entid(job)
        response = self._make_request('POST', f'/{self.endpoint_base}/cancel/{job_id}')
        return response.json().get('success', False)

    def list_active_jobs(self) -> dict:
        """List active deployment jobs count."""
        response = self._make_request('GET', f'/{self.endpoint_base}/jobs')
        return response.json()

    def list_images(self, model_name: str | None = None) -> list[dict]:
        """List deployed model images."""
        params = {}
        if model_name:
            params['model_name'] = model_name
        response = self._make_request('GET', f'/{self.endpoint_base}/images', params=params)
        return response.json()

    def remove_image(self, model_name: str, tag: str | None = None) -> dict:
        """Remove a deployed model image."""
        params = {}
        if tag:
            params['tag'] = tag
        response = self._make_request('DELETE', f'/{self.endpoint_base}/image/{model_name}', params=params)
        return response.json()

    def image_exists(self, model_name: str, tag: str = "champion") -> bool:
        """Check if a model image exists."""
        params = {'tag': tag}
        response = self._make_request('GET', f'/{self.endpoint_base}/image/{model_name}/exists', params=params)
        return response.json().get('exists', False)

