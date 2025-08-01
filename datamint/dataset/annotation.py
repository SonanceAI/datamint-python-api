from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
from PIL import Image
import json

# if TYPE_CHECKING:
#     from datamint.apihandler.annotation_api_handler import AnnotationAPIHandler

_LOGGER = logging.getLogger(__name__)


# Map API field names to class attributes
_FIELD_MAPPING = {
    'type':'annotation_type',
    'name': 'identifier',
    'added_by': 'created_by',
    'index': 'frame_index',
}

@dataclass
class Annotation:
    """
    Class representing an annotation from the Datamint API.
    
    This class stores annotation data and provides methods for loading 
    and saving annotations through the API handler.
    
    Args:
        id: Unique identifier for the annotation
        identifier: The annotation identifier/label name
        scope: Whether annotation applies to 'frame' or 'image'
        annotation_type: Type of annotation ('segmentation', 'label', 'category', etc.)
        resource_id: ID of the resource this annotation belongs to
        annotation_worklist_id: ID of the annotation worklist
        created_by: Email of the user who created the annotation
        status: Status of the annotation ('published', 'new', etc.)
        frame_index: Frame index for frame-scoped annotations
        text_value: Text value for category annotations
        numeric_value: Numeric value for numeric annotations
        units: Units for numeric annotations
        geometry: Geometry data for geometric annotations
        created_at: When the annotation was created
        approved_at: When the annotation was approved
        approved_by: Who approved the annotation
        associated_file: Path to associated file (for segmentations)
        deleted: Whether the annotation is deleted
        deleted_at: When the annotation was deleted
        deleted_by: Who deleted the annotation
        created_by_model: Model ID if created by AI
        old_geometry: Previous geometry data
        set_name: Set name for grouped annotations
        resource_filename: Filename of the associated resource
        resource_modality: Modality of the associated resource
        annotation_worklist_name: Name of the annotation worklist
        user_info: Information about the user who created the annotation
        values: Additional values
    """
    
    id: str
    identifier: str
    scope: str
    annotation_type: str
    resource_id: str
    created_by: str
    annotation_worklist_id: Optional[str] = None
    status: Optional[str] = None
    frame_index: Optional[int] = None
    text_value: Optional[str] = None
    numeric_value: Optional[float] = None
    units: Optional[str] = None
    geometry: list[Any] = field(default_factory=list)
    created_at: Optional[str] = None
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None
    associated_file: Optional[str] = None
    file: Optional[str] = None
    deleted: bool = False
    deleted_at: Optional[str] = None
    deleted_by: Optional[str] = None
    created_by_model: Optional[str] = None
    old_geometry: Optional[Any] = None
    set_name: Optional[str] = None
    resource_filename: Optional[str] = None
    resource_modality: Optional[str] = None
    annotation_worklist_name: Optional[str] = None
    user_info: Optional[dict[str, str]] = None
    values: Optional[Any] = None
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        """
        Create an Annotation instance from a dictionary.
        
        Args:
            data: Dictionary containing annotation data from API
            
        Returns:
            Annotation instance
        """


        # Convert field names and filter valid fields
        converted_data = {}
        for key, value in data.items():
            # Map field names if needed
            mapped_key = _FIELD_MAPPING.get(key, key)
            converted_data[mapped_key] = value

        if 'scope' not in converted_data:
            converted_data['scope'] = 'image' if converted_data.get('frame_index') is None else 'frame'

        if converted_data['annotation_type'] in ['segmentation']:
            if converted_data.get('file') is None:
                raise ValueError(f"Segmentation annotations must have an associated file. {data}")
        
        # Create instance with only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in converted_data.items() if k in valid_fields}

        return cls(**filtered_data)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the annotation to a dictionary format.
        
        Returns:
            Dictionary representation of the annotation
        """
        result = {}
        for key, value in self.__dict__.items():
            # Handle special serialization cases
            if isinstance(value, (np.ndarray, np.generic)):
                value = value.tolist()
            elif isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, Path):
                value = str(value)
            
            result[key] = value
        if self.annotation_type == 'segmentation' and 'file' not in result:
            raise ValueError(f"Segmentation annotations must have an associated file. {self}")
        return result

    @property
    def name(self) -> str:
        """Get the annotation name (alias for identifier)."""
        return self.identifier
    
    @property
    def type(self) -> str:
        """Get the annotation type."""
        return self.annotation_type
    
    @property
    def value(self) -> Optional[str]:
        """Get the annotation value (for category annotations)."""
        return self.text_value
    
    @property
    def index(self) -> Optional[int]:
        """Get the frame index (alias for frame_index)."""
        return self.frame_index
    
    @property
    def added_by(self) -> str:
        """Get the creator email (alias for created_by)."""
        return self.created_by
    
    # @property
    # def file(self) -> Optional[str]:
    #     """Get the associated file path."""
    #     return self.associated_file
    
    # @file.setter
    # def file(self, value: Optional[str]) -> None:
    #     """Set the associated file path."""
    #     self.associated_file = value
    
    def is_segmentation(self) -> bool:
        """Check if this is a segmentation annotation."""
        return self.annotation_type == 'segmentation'
    
    def is_label(self) -> bool:
        """Check if this is a label annotation."""
        return self.annotation_type == 'label'
    
    def is_category(self) -> bool:
        """Check if this is a category annotation."""
        return self.annotation_type == 'category'
    
    def is_frame_scoped(self) -> bool:
        """Check if this annotation is frame-scoped."""
        return self.scope == 'frame'
    
    def is_image_scoped(self) -> bool:
        """Check if this annotation is image-scoped."""
        return self.scope == 'image'
    
    def get_created_datetime(self) -> Optional[datetime]:
        """
        Get the creation datetime as a datetime object.
        
        Returns:
            datetime object or None if created_at is not set
        """
        if self.created_at:
            try:
                return datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
            except ValueError:
                _LOGGER.warning(f"Could not parse created_at datetime: {self.created_at}")
        return None
    
    def __repr__(self) -> str:
        """String representation of the annotation."""
        return (f"Annotation(id='{self.id}', identifier='{self.identifier}', "
                f"type='{self.annotation_type}', scope='{self.scope}', resource_id='{self.resource_id}')")
