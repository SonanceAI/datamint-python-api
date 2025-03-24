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

"""
Example:
    {
        "id": 0,
        "value": {},
        "type": "segmentation",
        "identifier": "string",
        "scope": "frame",
        "frame_index": 0,
        "annotation_worklist_id": "string",
        "units": "string",
        "geometry": {
            "boxes": [
                {
                    "x0": 0,
                    "y0": 0,
                    "x1": 0,
                    "y1": 0,
                    "frame_index": 0
                }
            ],
            "points": [
                {
                    "x": 0,
                    "y": 0,
                    "is_foreground": true,
                    "frame_index": 0
                }
            ],
            "external_description": {
                "data": {
                    "label": "",
                    "handles": {
                        "points": [
                            [
                                308.5447971781305,
                                203.11948853615525,
                                0
                            ],
                            [
                                310.8699294532628,
                                386.0298941798942,
                                0
                            ]
                        ],
                        "textBox": {
                            "hasMoved": false,
                            "worldPosition": [
                                310.8699294532628,
                                294.57469135802467,
                                0
                            ],
                            "worldBoundingBox": {
                                "topLeft": [
                                    330.24603174603175,
                                    313.9507936507936,
                                    0
                                ],
                                "topRight": [
                                    390.81572869585096,
                                    313.9507936507936,
                                    0
                                ],
                                "bottomLeft": [
                                    330.24603174603175,
                                    346.1151234567901,
                                    0
                                ],
                                "bottomRight": [
                                    390.81572869585096,
                                    346.1151234567901,
                                    0
                                ]
                            }
                        },
                        "activeHandleIndex": null
                    },
                    "cachedStats": {}
                },
                "isLocked": false,
                "metadata": {
                    "viewUp": [
                        0,
                        -1,
                        0
                    ],
                    "toolName": "Length",
                    "sliceIndex": 3,
                    "cameraPosition": [
                        400,
                        300,
                        -549.2300110154215
                    ],
                    "viewPlaneNormal": [
                        0,
                        0,
                        -1
                    ],
                    "cameraFocalPoint": [
                        400,
                        300,
                        0
                    ],
                    "referencedImageId": "wadouri:https://stagingapi.datamint.io/resources/660001c6-fd28-4081-97cf-d1f9597430f8/file?frame=4"
                },
                "isVisible": true,
                "isSelected": true,
                "highlighted": false,
                "invalidated": false,
                "annotationUID": "fc4a73c6-5b4f-4a5f-8479-2c377b19dc70"
            }
        },
        "from_frame": 0,
        "is_model": true,
        "model_id": "string",
        "import_author": "string",
        "imported_from": "string",
        "project_id": "string"
    }
"""


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
