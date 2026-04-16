from __future__ import annotations

from typing import ClassVar

from .volume_resource import VolumeResource


class NiftiResource(VolumeResource):
    """Represents a NIfTI volume resource."""

    resource_kind: ClassVar[str] = 'nifti'
    resource_priority: ClassVar[int] = 50
    storage_aliases: ClassVar[tuple[str, ...]] = ('NiftiResource', 'NiftiResourceHandler')
    mimetypes: ClassVar[tuple[str, ...]] = ('application/nifti',)
    filename_suffixes: ClassVar[tuple[str, ...]] = ('.nii', '.nii.gz')

    @property
    def is_compressed(self) -> bool:
        """Whether the underlying NIfTI file is gzip-compressed."""
        return self.filename.casefold().endswith('.nii.gz')