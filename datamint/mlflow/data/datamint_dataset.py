"""MLflow Dataset adapter for Datamint project splits."""
from __future__ import annotations

import hashlib
import json
from typing import Any
from collections.abc import Sequence
import logging

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from datamint.entities.resource import Resource


_LOGGER = logging.getLogger(__name__)


class DatamintDatasetSource(DatasetSource):
    """Source info pointing to a Datamint project."""

    def __init__(self, project_id: str, project_name: str,
                 split: str | None,
                 extra_params: dict[str, Any] | None = None) -> None:
        self._project_id = project_id
        self._project_name = project_name
        self._split = split
        self.extra_params = extra_params

    @staticmethod
    def _get_source_type() -> str:
        return "datamint"

    def load(self, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "DatamintDatasetSource.load() is not supported. "
            "Use the Datamint API to load data."
        )

    @staticmethod
    def _can_resolve(raw_source: str) -> bool:
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> DatamintDatasetSource:
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps({
            "project_id": self._project_id,
            "project_name": self._project_name,
            "split": self._split,
            "extra_params": self.extra_params,
        })

    @classmethod
    def from_json(cls, source_json: str) -> DatamintDatasetSource:
        data = json.loads(source_json)
        return cls(
            project_id=data["project_id"],
            project_name=data["project_name"],
            split=data["split"],
            extra_params=data.get("extra_params"),
        )


class DatamintMLflowDataset(Dataset):
    """MLflow Dataset wrapping a Datamint project split for lineage tracking."""

    def __init__(
        self,
        project_id: str,
        project_name: str,
        split: str | None,
        resources: Sequence[str] | Sequence[Resource],
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        self.resources = resources
        self.extra_params = extra_params
        source = DatamintDatasetSource(project_id, project_name, split,
                                       extra_params=extra_params)
        super().__init__(source=source, name=project_name)

    def _compute_digest(self) -> str:
        dumped_resources = []
        for r in self.resources:
            if isinstance(r, str):
                dumped_resources.append({'id': r})
            else:
                data = {
                    "id": r.id,
                    "obj_type": str(type(r)),
                }
                for attrname in ("slice_index", "slice_axis", "filename"):
                    a = getattr(r, attrname, None)
                    if a is not None:
                        data[attrname] = a
                dumped_resources.append(data)

        data: dict = {"resources": dumped_resources}
        if self.extra_params:
            data["extra_params"] = self.extra_params

        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]

    @property
    def profile(self) -> Any | None:
        storage_types = [r.storage if isinstance(r, Resource) else None for r in self.resources]
        most_common_storage_type = None
        if storage_types:
            most_common_storage_type = max(set(storage_types), key=storage_types.count)

        _LOGGER.debug(f"Computed profile for DatamintDataset with {len(self.resources)} resources. "
                      f"Most common storage type: {most_common_storage_type}")

        return {"num_resources": len(self.resources),
                "most_common_storage_type": most_common_storage_type}

    def to_dict(self) -> dict[str, str]:
        config = super().to_dict()
        config.update(
            {
                "schema": None,
                "profile": json.dumps(self.profile),
            }
        )
        return config

    @property
    def schema(self):
        return None
