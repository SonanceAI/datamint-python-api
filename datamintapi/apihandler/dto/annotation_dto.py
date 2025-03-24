import requests
import json

# export enum AnnotationType {
#     SEGMENTATION = 'segmentation',
#     AREA = 'area',
#     DISTANCE = 'distance',
#     ANGLE = 'angle',
#     POINT = 'point',
#     LINE = 'line',
#     REGION = 'region',
#     SQUARE = 'square',
#     CIRCLE = 'circle',
#     CATEGORY = 'category',
#     LABEL = 'label',
# }


class Box:
    def __init__(self, x0, y0, x1, y1, frame_index):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.frame_index = frame_index


class Handles:
    def __init__(self, points: list[list[float]]):
        self.points = points

    def to_dict(self) -> dict:
        return {
            "points": self.points,
            "textBox": None,
            "activeHandleIndex": None,
            "textBox": {
                "hasMoved": False,
            },
            "activeHandleIndex": None,
        }


class ExternalDescription:
    class Metadata:
        def __init__(self, sliceIndex: int,
                     referencedImageId: str,
                     toolName: str,
                     viewUp: list[int] = None,
                     cameraPosition: list[float] = None,
                     viewPlaneNormal: list[int] = None, cameraFocalPoint: list[float] = None,
                     ):
            self.viewUp = viewUp
            self.toolName = toolName
            self.sliceIndex = sliceIndex
            self.cameraPosition = cameraPosition
            self.viewPlaneNormal = viewPlaneNormal
            self.cameraFocalPoint = cameraFocalPoint
            self.referencedImageId = referencedImageId

        def to_dict(self) -> dict:
            return {
                "viewUp": self.viewUp,
                "toolName": self.toolName,
                "sliceIndex": self.sliceIndex,
                "cameraPosition": self.cameraPosition,
                "viewPlaneNormal": self.viewPlaneNormal,
                "cameraFocalPoint": self.cameraFocalPoint,
                "referencedImageId": self.referencedImageId
            }

    def __init__(self,
                 annotationUID: str,
                 label: str, handles: Handles,
                 metadata: Metadata,
                 highlighted: bool = False, invalidated: bool = False, isLocked: bool = False,
                 isVisible: bool = True,):
        self.annotationUID = annotationUID
        self.data = {
            "label": label,
            "handles": handles.to_dict(),
            "cachedStats": {}
        }
        self.highlighted = highlighted
        self.invalidated = invalidated
        self.isLocked = isLocked
        self.isVisible = isVisible
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "data": self.data,
            "highlighted": self.highlighted,
            "invalidated": self.invalidated,
            "isLocked": self.isLocked,
            "isVisible": self.isVisible,
            "metadata": self.metadata.to_dict(),
        }


class SamGeometry:
    def __init__(self, boxes: list[Box] | None = None, points=None,
                 ):
        self.boxes = boxes if boxes else None
        self.points = points if points else None

    def to_dict(self) -> dict:
        return {
            "boxes": [box.__dict__ for box in self.boxes] if self.boxes else None,
            "points": [point.__dict__ for point in self.points] if self.points else None,
        }


class MainGeometry:
    def __init__(self,
                 sam_geometry: SamGeometry,
                 external_description: ExternalDescription):
        self.sam_geometry = sam_geometry
        self.external_description = external_description

    def to_dict(self) -> dict:
        sam_geometry_dict = self.sam_geometry.to_dict() if self.sam_geometry else None
        ret = {
            "external_description": self.external_description.to_dict()
        }
        if sam_geometry_dict:
            for key in sam_geometry_dict:
                ret[key] = sam_geometry_dict[key]
        return ret


class CreateAnnotationDto:
    def __init__(self,
                 value, type: str, identifier: str,
                 scope: str,
                 annotation_worklist_id: str, imported_from: str, import_author: str,
                 frame_index: int, model_id: str, is_model: bool,
                 geometry: MainGeometry,
                 units: str = None):
        self.value = value
        self.type = type
        self.identifier = identifier
        self.scope = scope
        self.geometry = geometry
        self.annotation_worklist_id = annotation_worklist_id
        self.imported_from = imported_from
        self.import_author = import_author
        self.frame_index = frame_index
        self.units = units
        self.model_id = model_id
        self.is_model = is_model

    def to_dict(self):
        return {
            "value": self.value,
            "type": self.type,
            "identifier": self.identifier,
            "scope": self.scope,
            'frame_index': self.frame_index,
            'annotation_worklist_id': self.annotation_worklist_id,
            'imported_from': self.imported_from,
            'import_author': self.import_author,
            'units': self.units,
            "geometry": self.geometry.to_dict() if self.geometry else None
        }
