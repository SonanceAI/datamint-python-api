"""Deprecated: Use MLFlow API instead."""
from typing import Sequence
from ..entity_base_api import ApiConfig, BaseApi
import httpx
from datamint.exceptions import EntityAlreadyExistsError


class ModelsApi(BaseApi):
    """API handler for project-related endpoints."""

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None) -> None:
        """Initialize the projects API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        super().__init__(config, client=client)

    def create(self,
               name: str) -> dict:
        json = {
            'name': name
        }

        try:
            response = self._make_request('POST',
                                          'ai-models',
                                          json=json)
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                raise EntityAlreadyExistsError('ai-model', {'name': name})
            raise

    def get_all(self) -> Sequence[dict]:
        response = self._make_request('GET',
                                      'ai-models')
        return response.json()

    def get_by_name(self, name: str) -> dict | None:
        models = self.get_all()
        for model in models:
            if model['name'] == name:
                return model
        return None
