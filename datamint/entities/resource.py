"""Resource entity module for DataMint API."""

from datetime import datetime
from typing import Optional, Any
import logging
from .base_entity import BaseEntity, MISSING_FIELD
from pydantic import Field

logger = logging.getLogger(__name__)

class Resource(BaseEntity):
    """Represents a DataMint resource with all its properties and metadata.
    
    This class models a resource entity from the DataMint API, containing
    information about uploaded files, their metadata, and associated projects.
    
    Attributes:
        id: Unique identifier for the resource
        resource_uri: URI path to access the resource file
        storage: Storage type (e.g., 'DicomResource')
        location: Storage location path
        upload_channel: Channel used for upload (e.g., 'tmp')
        filename: Original filename of the resource
        modality: Medical imaging modality
        mimetype: MIME type of the file
        size: File size in bytes
        upload_mechanism: Mechanism used for upload (e.g., 'api')
        customer_id: Customer/organization identifier
        status: Current status of the resource
        created_at: ISO timestamp when resource was created
        created_by: Email of the user who created the resource
        published: Whether the resource is published
        published_on: ISO timestamp when resource was published
        published_by: Email of the user who published the resource
        publish_transforms: Optional publication transforms
        deleted: Whether the resource is deleted
        deleted_at: Optional ISO timestamp when resource was deleted
        deleted_by: Optional email of the user who deleted the resource
        metadata: Resource metadata with DICOM information
        source_filepath: Original source file path
        tags: List of tags associated with the resource
        instance_uid: DICOM SOP Instance UID (top-level)
        series_uid: DICOM Series Instance UID (top-level)
        study_uid: DICOM Study Instance UID (top-level)
        patient_id: Patient identifier (top-level)
        segmentations: Optional segmentation data
        measurements: Optional measurement data
        categories: Optional category data
        labels: List of labels associated with the resource
        user_info: Information about the user who created the resource
        projects: List of projects this resource belongs to
    """
    id: str
    resource_uri: str
    storage: str
    location: str
    upload_channel: str
    filename: str
    modality: str
    mimetype: str
    size: int
    upload_mechanism: str
    customer_id: str
    status: str
    created_at: str
    created_by: str
    published: bool
    deleted: bool
    source_filepath: str | None
    metadata: dict
    projects: list[dict] = MISSING_FIELD
    published_on: str | None
    published_by: str | None
    tags: list[str] | None = None
    publish_transforms: Optional[Any] = None
    deleted_at: Optional[str] = None
    deleted_by: Optional[str] = None
    instance_uid: Optional[str] = None
    series_uid: Optional[str] = None
    study_uid: Optional[str] = None
    patient_id: Optional[str] = None
    segmentations: Optional[Any] = None  # TODO: Define proper type when spec available
    measurements: Optional[Any] = None  # TODO: Define proper type when spec available
    categories: Optional[Any] = None  # TODO: Define proper type when spec available
    user_info: Optional[dict] = None

    @property
    def size_mb(self) -> float:
        """Get file size in megabytes.
        
        Returns:
            File size in MB rounded to 2 decimal places
        """
        return round(self.size / (1024 * 1024), 2)
    
    def is_dicom(self) -> bool:
        """Check if the resource is a DICOM file.
        
        Returns:
            True if the resource is a DICOM file, False otherwise
        """
        return self.mimetype == 'application/dicom' or self.storage == 'DicomResource'
    
    def get_project_names(self) -> list[str]:
        """Get list of project names this resource belongs to.
        
        Returns:
            List of project names
        """
        return [proj['name'] for proj in self.projects]
    
    def __str__(self) -> str:
        """String representation of the resource.
        
        Returns:
            Human-readable string describing the resource
        """
        return f"Resource(id='{self.id}', filename='{self.filename}', size={self.size_mb}MB)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the resource.
        
        Returns:
            Detailed string representation for debugging
        """
        return (
            f"Resource(id='{self.id}', filename='{self.filename}', "
            f"modality='{self.modality}', status='{self.status}', "
            f"published={self.published})"
        )
