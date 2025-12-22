"""Cache manager for storing and retrieving entity-related data locally.

This module provides caching functionality for resource data (images, segmentations, etc.)
with automatic validation against server versions to ensure data freshness.
"""

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, Generic
from pydantic import BaseModel
# import appdirs
import datamint.configs

_LOGGER = logging.getLogger(__name__)

T = TypeVar('T')


class CacheManager(Generic[T]):
    """Manages local caching of entity data with versioning support.

    This class handles storing and retrieving cached data with automatic
    validation against server versions to ensure data consistency.

    The cache uses a directory structure:
    - cache_root/
      - resources/
        - {resource_id}/
          - image_data.pkl
          - metadata.json
      - annotations/
        - {annotation_id}/
          - segmentation_data.pkl
          - metadata.json

    Attributes:
        cache_root: Root directory for cache storage
        entity_type: Type of entity being cached (e.g., 'resources', 'annotations')
    """

    class ItemMetadata(BaseModel):
        cached_at: datetime
        data_path: str
        data_type: str
        mimetype: str
        version_hash: str | None = None
        version_info: dict | None = None
        entity_id: str | None = None

    def __init__(self, entity_type: str, cache_root: Path | str | None = None):
        """Initialize the cache manager.

        Args:
            entity_type: Type of entity (e.g., 'resources', 'annotations')
            cache_root: Root directory for cache. If None, uses system cache directory.
        """
        self.entity_type = entity_type

        if cache_root is None:
            # Use platform-specific cache directory
            # app_cache_dir = appdirs.user_cache_dir('datamint', 'sonance')
            # cache_root = Path(app_cache_dir) / 'entity_cache'
            cache_root = Path(datamint.configs.DATAMINT_DATA_DIR)
        else:
            cache_root = Path(cache_root)

        self.cache_root = cache_root / entity_type

    def _get_entity_cache_dir(self, entity_id: str) -> Path:
        """Get the cache directory for a specific entity.

        Args:
            entity_id: Unique identifier for the entity

        Returns:
            Path to the entity's cache directory
        """
        entity_dir = self.cache_root / entity_id
        entity_dir = entity_dir.resolve().absolute()
        entity_dir.mkdir(parents=True, exist_ok=True)
        return entity_dir

    def _get_metadata_path(self, entity_id: str) -> Path:
        """Get the path to the metadata file for an entity.

        Args:
            entity_id: Unique identifier for the entity

        Returns:
            Path to the metadata file
        """
        return self._get_entity_cache_dir(entity_id) / 'metadata.json'

    def _get_data_path(self, entity_id: str, data_key: str) -> Path:
        """Get the path to a data file for an entity.

        Args:
            entity_id: Unique identifier for the entity
            data_key: Key identifying the type of data (e.g., 'image_data', 'segmentation')

        Returns:
            Path to the data file
        """

        datapath = self._get_entity_cache_dir(entity_id) / f"{data_key}"
        if datapath.with_suffix('.pkl').exists():
            return datapath.with_suffix('.pkl')
        return datapath

    def _compute_version_hash(self, version_info: dict[str, Any]) -> str:
        """Compute a hash from version information.

        Args:
            version_info: Dictionary containing version information (e.g., updated_at, size)

        Returns:
            Hash string representing the version
        """
        # Sort keys for consistent hashing
        sorted_info = json.dumps(version_info, sort_keys=True)
        return hashlib.sha256(sorted_info.encode()).hexdigest()

    def _get_validated_metadata(
        self,
        entity_id: str,
        data_key: str,
        version_info: dict[str, Any] | None = None
    ) -> tuple['CacheManager.ItemMetadata', Path] | tuple[None, None]:
        """Get and validate cached metadata for an entity.

        Args:
            entity_id: Unique identifier for the entity
            data_key: Key identifying the type of data
            version_info: Optional version information from server to validate cache

        Returns:
            Tuple of (metadata, data_path) if valid, (None, None) if cache miss or invalid
        """
        metadata_path = self._get_metadata_path(entity_id)
        data_path = self._get_data_path(entity_id, data_key)

        if not metadata_path.exists() or not data_path.exists():
            _LOGGER.debug(f"Cache miss for {entity_id}/{data_key}")
            return None, None

        try:
            with open(metadata_path, 'r') as f:
                jsondata = f.read()
            cached_metadata = CacheManager.ItemMetadata.model_validate_json(jsondata)

            if version_info is not None:
                server_version = self._compute_version_hash(version_info)
                if server_version != cached_metadata.version_hash:
                    _LOGGER.debug(
                        f"Cache version mismatch for {entity_id}/{data_key}. "
                        f"Server: {server_version}, Cached: {cached_metadata.version_hash}"
                    )
                    return None, None

            return cached_metadata, data_path
        except Exception as e:
            _LOGGER.warning(f"Error reading cache metadata for {entity_id}/{data_key}: {e}")
            return None, None

    def get(
        self,
        entity_id: str,
        data_key: str,
        version_info: dict[str, Any] | None = None
    ) -> T | None:
        """Retrieve cached data for an entity.

        Args:
            entity_id: Unique identifier for the entity
            data_key: Key identifying the type of data
            version_info: Optional version information from server to validate cache

        Returns:
            Cached data if valid, None if cache miss or invalid
        """
        cached_metadata, data_path = self._get_validated_metadata(entity_id, data_key, version_info)
        
        if cached_metadata is None:
            return None

        try:
            data = self._load_data(cached_metadata)
            _LOGGER.debug(f"Cache hit for {entity_id}/{data_key}")
            return data
        except Exception as e:
            _LOGGER.warning(f"Error reading cache for {entity_id}/{data_key}: {e}")
            return None

    def get_path(
        self,
        entity_id: str,
        data_key: str,
        version_info: dict[str, Any] | None = None
    ) -> Path | None:
        """Get the path to cached data for an entity if valid.

        Args:
            entity_id: Unique identifier for the entity
            data_key: Key identifying the type of data
            version_info: Optional version information from server to validate cache

        Returns:
            Path to cached data if valid, None if cache miss or invalid
        """
        cached_metadata, data_path = self._get_validated_metadata(entity_id, data_key, version_info)
        return data_path

    def set(
        self,
        entity_id: str,
        data_key: str,
        data: T,
        version_info: dict[str, Any] | None = None
    ) -> None:
        """Store data in cache for an entity.

        Args:
            entity_id: Unique identifier for the entity
            data_key: Key identifying the type of data
            data: Data to cache
            version_info: Optional version information from server
        """
        metadata_path = self._get_metadata_path(entity_id)
        data_path = self._get_data_path(entity_id, data_key)

        try:
            mimetype = self._save_data(data_path, data)

            metadata = CacheManager.ItemMetadata(
                cached_at=datetime.now(),
                data_path=str(data_path.absolute()),
                data_type=type(data).__name__,
                mimetype=mimetype,
                entity_id=entity_id
            )

            # Update metadata for this data key

            if version_info is not None:
                metadata.version_hash = self._compute_version_hash(version_info)
                # Store version_info as JSON string to ensure metadata is JSON-serializable
                metadata.version_info = version_info

            # Save metadata
            with open(metadata_path, 'w') as f:
                f.write(metadata.model_dump_json(indent=2))

            _LOGGER.debug(f"Cached data for {entity_id}/{data_key}")

        except Exception as e:
            _LOGGER.warning(f"Error writing cache for {entity_id}/{data_key}: {e}")

    def _load_data(self,
                   metadata: 'CacheManager.ItemMetadata') -> T:
        path = metadata.data_path
        if metadata.mimetype == 'application/octet-stream':
            with open(path, 'rb') as f:
                return f.read()
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)


    def _save_data(self, path: Path, data: T) -> str:
        """
        Save data and returns the mimetype
        """
        if isinstance(data, bytes):
            with open(path, 'wb') as f:
                f.write(data)
            return 'application/octet-stream'
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            return 'application/x-python-serialize'

    def invalidate(self, entity_id: str, data_key: str | None = None) -> None:
        """Invalidate cached data for an entity.

        Args:
            entity_id: Unique identifier for the entity
            data_key: Optional key for specific data. If None, invalidates all data for entity.
        """
        if data_key is None:
            # Invalidate entire entity cache
            entity_dir = self._get_entity_cache_dir(entity_id)
            if entity_dir.exists():
                import shutil
                shutil.rmtree(entity_dir)
                _LOGGER.debug(f"Invalidated all cache for {entity_id}")
        else:
            # Invalidate specific data
            data_path = self._get_data_path(entity_id, data_key)
            if data_path.exists():
                data_path.unlink()
                _LOGGER.debug(f"Invalidated cache for {entity_id}/{data_key}")

            # Update metadata
            metadata_path = self._get_metadata_path(entity_id)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                if data_key in metadata:
                    del metadata[data_key]

                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

    def clear_all(self) -> None:
        """Clear all cached data for this entity type."""
        if self.cache_root.exists():
            import shutil
            shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(parents=True, exist_ok=True)
            _LOGGER.info(f"Cleared all cache for {self.entity_type}")

    def get_cache_info(self, entity_id: str) -> dict[str, Any]:
        """Get information about cached data for an entity.

        Args:
            entity_id: Unique identifier for the entity

        Returns:
            Dictionary containing cache information
        """
        metadata_path = self._get_metadata_path(entity_id)

        if not metadata_path.exists():
            return {}

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            _LOGGER.warning(f"Error reading cache info for {entity_id}: {e}")
            return {}
