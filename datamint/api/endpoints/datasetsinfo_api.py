from ..entity_base_api import ApiConfig, EntityBaseApi
from datamint.entities.datasetinfo import DatasetInfo
import httpx


class DatasetsInfoApi(EntityBaseApi[DatasetInfo]):
    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None) -> None:
        """Initialize the datasets API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        super().__init__(config, DatasetInfo, 'datasets', client)
