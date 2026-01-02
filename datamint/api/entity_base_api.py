from typing import Any, Literal, TypeVar, Generic, overload
from collections.abc import Sequence, AsyncGenerator
import logging
import httpx
from datamint.entities.base_entity import BaseEntity
from datamint.exceptions import DatamintException, ResourceNotFoundError
import aiohttp
import asyncio
from .base_api import ApiConfig, BaseApi
import contextlib

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseEntity)


class EntityBaseApi(BaseApi, Generic[T]):
    """Base API handler for entity-related endpoints with CRUD operations.

    This class provides a template for API handlers that work with specific
    entity types, offering common CRUD operations with proper typing.

    Type Parameters:
        T: The entity type this API handler manages (must extend BaseEntity)
    """

    def __init__(self, config: ApiConfig,
                 entity_class: type[T],
                 endpoint_base: str,
                 client: httpx.Client | None = None
                 ) -> None:
        """Initialize the entity API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            entity_class: The entity class this handler manages
            endpoint_base: Base endpoint path (e.g., 'projects', 'annotations')
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        super().__init__(config, client)
        self.__entity_class = entity_class
        self.endpoint_base = endpoint_base.strip('/')

    def _init_entity_obj(self, **kwargs) -> T:
        obj = self.__entity_class(**kwargs)
        obj._api = self
        return obj

    @staticmethod
    def _entid(entity: BaseEntity | str) -> str:
        return entity if isinstance(entity, str) else entity.id

    def _make_entity_request(self,
                             method: str,
                             entity_id: str | BaseEntity,
                             add_path: str = '',
                             **kwargs) -> httpx.Response:
        """
        Make an HTTP request for a specific entity by its ID.
        It is basically a wrapper around :py:meth:`_make_request` that
        constructs the URL for the entity in this form: `/{endpoint_base}/{entity_id}/{add_path}`
        """
        try:
            entity_id = self._entid(entity_id)
            add_path = '/'.join(add_path.strip().strip('/').split('/'))
            return self._make_request(method, f'/{self.endpoint_base}/{entity_id}/{add_path}', **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ResourceNotFoundError(self.endpoint_base, {'id': entity_id}) from e
            raise
        except ResourceNotFoundError as e:
            e.resource_type = self.endpoint_base
            e.params = {'id': entity_id}
            raise

    @contextlib.asynccontextmanager
    async def _make_entity_request_async(self,
                                         method: str,
                                         entity_id: str | BaseEntity,
                                         add_path: str = '',
                                         session: aiohttp.ClientSession | None = None,
                                         **kwargs) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        try:
            entity_id = self._entid(entity_id)
            add_path = '/'.join(add_path.strip().strip('/').split('/'))
            async with self._make_request_async(method,
                                                f'/{self.endpoint_base}/{entity_id}/{add_path}',
                                                session=session,
                                                **kwargs) as resp:
                yield resp
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                raise ResourceNotFoundError(self.endpoint_base, {'id': entity_id}) from e
            raise
        except ResourceNotFoundError as e:
            e.resource_type = self.endpoint_base
            e.params = {'id': entity_id}
            raise

    async def _make_entity_request_async_json(self,
                                              method: str,
                                              entity_id: str | BaseEntity,
                                              add_path: str = '',
                                              session: aiohttp.ClientSession | None = None,
                                              **kwargs):
        async with self._make_entity_request_async(method,
                                                   entity_id,
                                                   add_path=add_path,
                                                   session=session,
                                                   **kwargs) as resp:
            return await resp.json()

    def _stream_entity_request(self,
                               method: str,
                               entity_id: str,
                               add_path: str = '',
                               **kwargs):
        try:
            add_path = '/'.join(add_path.strip().strip('/').split('/'))
            return self._stream_request(method, f'/{self.endpoint_base}/{entity_id}/{add_path}', **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ResourceNotFoundError(self.endpoint_base, {'id': entity_id}) from e
            raise

    def get_list(self, limit: int | None = None,
                 **kwargs) -> Sequence[T]:
        """Get entities with optional filtering.

        Returns:
            List of entity instances.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        new_kwargs = dict(kwargs)

        # Remove None values from the payload.
        for k in list(new_kwargs.keys()):
            if new_kwargs[k] is None:
                del new_kwargs[k]

        items_gen = self._make_request_with_pagination('GET', f'/{self.endpoint_base}',
                                                       return_field=self.endpoint_base,
                                                       limit=limit,
                                                       **new_kwargs)

        all_items = []
        for resp, items in items_gen:
            all_items.extend(items)

        return [self._init_entity_obj(**item) for item in all_items]

    def get_all(self, limit: int | None = None) -> Sequence[T]:
        """Get all entities with optional pagination and filtering.

        Returns:
            List of entity instances

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        return self.get_list(limit=limit)

    def get_by_id(self, entity_id: str) -> T:
        """Get a specific entity by its ID.

        Args:
            entity_id: Unique identifier for the entity.

        Returns:
            Entity instance.

        Raises:
            httpx.HTTPStatusError: If the entity is not found or request fails.
        """
        response = self._make_entity_request('GET', entity_id)
        return self._init_entity_obj(**response.json())

    async def _create_async(self, entity_data: dict[str, Any]) -> str | Sequence[str | dict]:
        """Create a new entity.

        Args:
            entity_data: Dictionary containing entity data for creation.

        Returns:
            The id of the created entity.

        Raises:
            httpx.HTTPStatusError: If creation fails.
        """
        respdata = await self._make_request_async_json('POST',
                                                       f'/{self.endpoint_base}',
                                                       json=entity_data)
        if 'error' in respdata:
            raise DatamintException(respdata['error'])
        if isinstance(respdata, str):
            return respdata
        if isinstance(respdata, list):
            return respdata
        if isinstance(respdata, dict):
            return respdata.get('id')
        return respdata

    def _get_child_entities(self,
                            parent_entity: BaseEntity | str,
                            child_entity_name: str) -> httpx.Response:
        response = self._make_entity_request('GET', parent_entity,
                                             add_path=child_entity_name)
        return response


class DeletableEntityApi(EntityBaseApi[T]):
    """Extension of EntityBaseApi for entities that support soft deletion.

    This class adds methods to handle soft-deleted entities, allowing
    retrieval and restoration of such entities.
    """

    def delete(self, entity: str | T) -> None:
        """Delete an entity.

        Args:
            entity: Unique identifier for the entity to delete or the entity instance itself.

        Raises:
            httpx.HTTPStatusError: If deletion fails or entity not found
        """
        self._make_entity_request('DELETE', entity)

    def bulk_delete(self, entities: Sequence[str | T]) -> None:
        """Delete multiple entities.

        Args:
            entities: Sequence of unique identifiers for the entities to delete or the entity instances themselves.

        Raises:
            httpx.HTTPStatusError: If deletion fails or any entity not found
        """
        async def _delete_all_async():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._delete_async(entity, session)
                    for entity in entities
                ]
                await asyncio.gather(*tasks)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_delete_all_async())

    async def _delete_async(self,
                            entity: str | BaseEntity,
                            session: aiohttp.ClientSession | None = None) -> None:
        """Asynchronously delete an entity by its ID.

        Args:
            entity: Unique identifier for the entity to delete or the entity instance itself.

        Raises:
            httpx.HTTPStatusError: If deletion fails or entity not found
        """
        async with self._make_entity_request_async('DELETE', entity,
                                                   session=session) as resp:
            await resp.text()  # Consume response to complete request

    # def get_deleted(self, **kwargs) -> Sequence[T]:
    #     pass

    # def restore(self, entity_id: str | BaseEntity) -> T:
    #     pass


class CreatableEntityApi(EntityBaseApi[T]):
    """Extension of EntityBaseApi for entities that support creation.

    This class adds methods to handle creation of new entities.
    """

    @overload
    def _create(self, entity_data: dict[str, Any],
                return_entity: Literal[True] = True) -> T | list[T]: ...

    @overload
    def _create(self, entity_data: dict[str, Any],
                return_entity: Literal[False]) -> str | list: ...

    def _create(self, entity_data: dict[str, Any],
                return_entity: bool = False) -> str | T | list:
        """Create a new entity.

        Args:
            entity_data: Dictionary containing entity data for creation.

        Returns:
            The id of the created entity.

        Raises:
            httpx.HTTPStatusError: If creation fails.
        """
        response = self._make_request('POST', f'/{self.endpoint_base}', json=entity_data)
        respdata = response.json()
        if isinstance(respdata, str):
            if return_entity:
                return self.get_by_id(respdata)
            return respdata
        if isinstance(respdata, list):
            if return_entity:
                logger.warning("Current implementation is slow when returning entities on bulk create."
                               " Try ``return_entity=False`` for better performance.")
                return [self.get_by_id(item['id']) if isinstance(item, dict) and 'id' in item else self.get_by_id(item)
                        for item in respdata]
            return respdata
        if isinstance(respdata, dict):
            if return_entity:
                try:
                    return self._init_entity_obj(**respdata)
                except:
                    logger.debug("Failed to init entity obj on create response. Falling back to get_by_id.")
                    return self.get_by_id(respdata.get('id'))
            return respdata.get('id')
        return respdata

    @overload
    def create(self, *args, return_entity: Literal[True] = True, **kwargs) -> T: ...

    @overload
    def create(self, *args, return_entity: Literal[False], **kwargs) -> str: ...

    def create(self, *args,
               return_entity: bool = True,
               **kwargs) -> str | T:
        raise NotImplementedError("Subclasses must implement the create method with their own custom parameters")


class UpdatableEntityApi(EntityBaseApi[T]):
    # def update(self, entity_id: str, entity_data: dict[str, Any]):
    #     """Update an existing entity.

    #     Args:
    #         entity_id: Unique identifier for the entity.
    #         entity_data: Dictionary containing updated entity data.

    #     Returns:
    #         Updated entity instance.

    #     Raises:
    #         httpx.HTTPStatusError: If update fails or entity not found.
    #     """
    #     self._make_entity_request('PUT', entity_id, json=entity_data)

    def patch(self, entity: str | T, entity_data: dict[str, Any]):
        """Partially update an existing entity.

        Args:
            entity: Unique identifier for the entity or the entity instance.
            entity_data: Dictionary containing fields to update. Only provided fields will be updated.

        Returns:
            Updated entity instance.

        Raises:
            httpx.HTTPStatusError: If update fails or entity not found.
        """
        self._make_entity_request('PATCH', entity, json=entity_data)

    def partial_update(self, entity: str | T, entity_data: dict[str, Any]):
        """Alias for :py:meth:`patch` to partially update an entity."""
        return self.patch(entity, entity_data)


class CRUDEntityApi(CreatableEntityApi[T], UpdatableEntityApi[T], DeletableEntityApi[T]):
    """Full CRUD API handler for entities supporting create, read, update, delete operations."""
    pass
