"""Local JSONL buffer for MLflow log entries that could not reach the remote store.

Used by `DatamintStore` to survive transient connection drops during training:
entries that fail to send are appended here instead of raising, and can be
replayed later via `replay_offline_logs()`.
"""
import json
import logging
import threading
from pathlib import Path
from typing import Any

import datamint.configs

_LOGGER = logging.getLogger(__name__)


def _default_buffer_root() -> Path:
    if datamint.configs.DATAMINT_DATA_DIR is None:
        raise RuntimeError(
            "No Datamint data directory available to store the offline MLflow buffer."
        )
    return Path(datamint.configs.DATAMINT_DATA_DIR) / 'offline_mlflow_buffer'


class OfflineLogBuffer:
    """Append-only local buffer of MLflow log entries for a single run."""

    def __init__(self, run_id: str, buffer_root: Path | str | None = None):
        root = Path(buffer_root) if buffer_root is not None else _default_buffer_root()
        self._run_dir = root / run_id
        self._file_path = self._run_dir / 'entries.jsonl'
        self._lock = threading.Lock()

    def append(self, kind: str, data: dict[str, Any]) -> None:
        """Append one log entry. `kind` is 'metric', 'param', or 'tag'."""
        entry = {"kind": kind, "data": data}
        with self._lock:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')

    def read_all(self) -> list[dict[str, Any]]:
        """Read every buffered entry, in the order they were appended."""
        with self._lock:
            if not self._file_path.exists():
                return []
            with open(self._file_path) as f:
                lines = f.readlines()

        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
        return entries

    def clear(self) -> None:
        """Delete all buffered entries for this run."""
        with self._lock:
            if self._file_path.exists():
                self._file_path.unlink()

    def is_empty(self) -> bool:
        return not self._file_path.exists() or self._file_path.stat().st_size == 0
