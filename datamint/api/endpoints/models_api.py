"""Deprecated: Use MLFlow API instead."""
from collections.abc import Sequence
from ..entity_base_api import BaseApi
import httpx
from datamint.exceptions import EntityAlreadyExistsError


class ModelsApi(BaseApi):
    """API handler for project-related endpoints."""

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
