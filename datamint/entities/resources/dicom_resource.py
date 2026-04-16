from __future__ import annotations

from typing import ClassVar

from .volume_resource import VolumeResource


class DICOMResource(VolumeResource):
    """Represents a DICOM resource or assembled DICOM series."""

    resource_kind: ClassVar[str] = 'dicom'
    resource_priority: ClassVar[int] = 50
    storage_aliases: ClassVar[tuple[str, ...]] = ('DicomResource', 'DicomResourceHandler')
    mimetypes: ClassVar[tuple[str, ...]] = ('application/dicom',)
    filename_suffixes: ClassVar[tuple[str, ...]] = ('.dcm',)

    @property
    def uids(self) -> dict[str, str]:
        """Return the available DICOM UIDs for this resource."""
        return {
            name: value
            for name, value in {
                'instance_uid': self.instance_uid,
                'series_uid': self.series_uid,
                'study_uid': self.study_uid,
            }.items()
            if value is not None
        }