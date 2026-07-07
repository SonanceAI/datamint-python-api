from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import DatamintBaseDataset


class SplitResult(dict):
    """A dict of split name → dataset that can persist itself to the server.

    Behaves exactly like a plain ``dict`` -- indexing, iteration, and all
    standard dict operations work unchanged.  The extra ``.save()`` method
    pushes the split assignments to the Datamint project so they become the
    official server-side split.

    Example::

        parts = dataset.split(train=0.8, test=0.2, by_patient=True, seed=42)

        parts['train']          # works as before
        for name, ds in parts.items(): ...  # works as before

        parts.save()            # persist to server
        parts.save(force=True)  # overwrite existing assignments
    """

    def save(self, force: bool = False) -> None:
        """Persist split assignments to the server.

        Args:
            force: If ``True``, overwrite any existing split assignments on
                the project.  If ``False`` (default) and the project already
                has assignments, a ``ValueError`` is raised.

        Raises:
            ValueError: If the dataset was not loaded from a project, or if
                the project already has split assignments and ``force=False``.
        """
        any_ds: DatamintBaseDataset = next(iter(self.values()))
        project = getattr(any_ds, 'project', None)
        if project is None:
            raise ValueError(
                "Cannot save splits: the dataset was not loaded from a project. "
                "Load with ImageDataset(project='...') or VolumeDataset(project='...') first."
            )

        api = any_ds._api
        existing = api.projects.get_splits(project)
        if existing and not force:
            raise ValueError(
                f"Project '{project.name}' already has split assignments. "
                "Use save(force=True) to overwrite."
            )

        for split_name, ds in self.items():
            api.projects.assign_splits(ds.resources, split_name, project=project)
