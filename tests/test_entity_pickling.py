from __future__ import annotations

import importlib
import inspect
import pickle
import pkgutil
from collections.abc import Callable

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image

import datamint.entities as entities_pkg
from datamint.entities.annotations import (
    Annotation,
    BoxAnnotation,
    BoxGeometry,
    ImageClassification,
    ImageSegmentation,
    LineAnnotation,
    LineGeometry,
    VolumeSegmentation,
)
from datamint.entities.annotations.annotation import AnnotationBase
from datamint.entities.annotations.base_geometry import BaseGeometryAnnotation
from datamint.entities.annotations.base_segmentation import BaseSegmentationAnnotation
from datamint.entities.base_entity import BaseEntity, BaseEntityModel
from datamint.entities.channel import Channel, ChannelResourceData
from datamint.entities.datasetinfo import DatasetInfo
from datamint.entities.deployjob import DeployJob
from datamint.entities.inferencejob import InferenceJob
from datamint.entities.project import Project
from datamint.entities.project_resource_split import ProjectResourceSplit
from datamint.entities.resource import LocalResource, Resource
from datamint.entities.resources import (
    DICOMResource,
    ImageResource,
    NiftiResource,
    VideoResource,
    VolumeResource,
)
from datamint.entities.sliced_resource import SlicedVolumeResource
from datamint.entities.sliced_video_resource import SlicedVideoResource
from datamint.entities.user import User

Factory = Callable[[], object]

_EXCLUDED_ENTITY_CLASSES = {
    AnnotationBase,
    BaseEntity,
    BaseEntityModel,
    BaseGeometryAnnotation,
    BaseSegmentationAnnotation,
}


def _resource_payload(**overrides) -> dict:
    payload = {
        'id': 'resource-1',
        'resource_uri': '/resources/resource-1/file',
        'storage': 'BinaryBlob',
        'location': 'resources/customer/resource-1',
        'upload_channel': 'test-channel',
        'filename': 'artifact.bin',
        'mimetype': 'application/octet-stream',
        'size': 128,
        'customer_id': 'customer-1',
        'status': 'inbox',
        'created_at': '2026-04-13T10:00:00Z',
        'created_by': 'tester@datamint.io',
        'published': False,
        'deleted': False,
        'upload_mechanism': 'api',
        'metadata': {},
        'modality': 'OT',
        'source_filepath': '/tmp/artifact.bin',
        'tags': ['pickle'],
        'user_info': {'firstname': 'Test', 'lastname': 'User'},
    }
    payload.update(overrides)
    return payload


def _make_annotation() -> Annotation:
    return Annotation(
        id='annotation-1',
        identifier='annotation',
        scope='image',
        annotation_type='category',
        text_value='present',
    )


def _make_image_classification() -> ImageClassification:
    return ImageClassification(id='classification-1', name='finding', value='positive')


def _make_image_segmentation() -> ImageSegmentation:
    return ImageSegmentation(
        id='image-segmentation-1',
        identifier='mask',
        segmentation_data=np.array([[0, 1], [1, 0]], dtype=np.uint8),
    )


def _make_volume_segmentation() -> VolumeSegmentation:
    segmentation = np.array(
        [
            [[0, 1], [1, 0]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.uint8,
    )
    return VolumeSegmentation.from_semantic_segmentation(
        segmentation,
        {1: 'tumor'},
        id='volume-segmentation-1',
    )


def _make_line_annotation() -> LineAnnotation:
    return LineAnnotation(
        id='line-1',
        identifier='measurement',
        geometry=LineGeometry(
            points=((1, 2, None), (3, 4, None)),
            coordinate_system='pixel',
        ),
    )


def _make_box_annotation() -> BoxAnnotation:
    return BoxAnnotation(
        id='box-1',
        identifier='lesion',
        geometry=BoxGeometry(
            points=((1, 2, None), (5, 6, None)),
            coordinate_system='pixel',
        ),
    )


def _make_channel() -> Channel:
    return Channel(
        id='channel-1',
        channel_name='uploads',
        resource_data=[
            ChannelResourceData(
                created_by='tester@datamint.io',
                customer_id='customer-1',
                resource_id='resource-1',
                resource_file_name='scan.png',
                resource_mimetype='image/png',
            )
        ],
    )


def _make_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        id='dataset-1',
        name='Dataset',
        created_at='2026-04-13T10:00:00Z',
        created_by='tester@datamint.io',
        description='Dataset for pickle tests',
        customer_id='customer-1',
        updated_at=None,
        total_resource=1,
        resource_ids=['resource-1'],
    )


def _make_deploy_job() -> DeployJob:
    return DeployJob(
        id='deploy-job-1',
        status='queued',
        model_name='classifier',
        build_logs='waiting',
    )


def _make_inference_job() -> InferenceJob:
    return InferenceJob(
        id='inference-job-1',
        status='running',
        model_name='classifier',
        resource_id='resource-1',
        progress_percentage=50,
    )


def _make_project() -> Project:
    return Project(
        id='project-1',
        name='Project',
        created_at='2026-04-13T10:00:00Z',
        created_by='tester@datamint.io',
        dataset_id='dataset-1',
        worklist_id='worklist-1',
        archived=False,
        resource_count=1,
        annotated_resource_count=0,
        description='Project for pickle tests',
        viewable_ai_segs=[],
        editable_ai_segs=[],
    )


def _make_project_resource_split() -> ProjectResourceSplit:
    return ProjectResourceSplit(
        id='split-1',
        split_name='train',
        project_id='project-1',
        resource_id='resource-1',
        created_at='2026-04-13T10:00:00Z',
        created_by='tester@datamint.io',
    )


def _make_user() -> User:
    return User(
        email='annotator@datamint.io',
        firstname='Annotator',
        lastname='User',
        roles=['annotator'],
        customer_id='customer-1',
        created_at='2026-04-13T10:00:00Z',
    )


def _make_resource() -> Resource:
    return Resource(**_resource_payload())


def _make_image_resource() -> ImageResource:
    return ImageResource(
        **_resource_payload(
            id='image-resource-1',
            storage='ImageResource',
            filename='scan.png',
            mimetype='image/png',
            metadata={'width': 32, 'height': 24},
            modality='US',
        )
    )


def _make_volume_resource() -> VolumeResource:
    return VolumeResource(
        **_resource_payload(
            id='volume-resource-1',
            storage='VolumeResource',
            filename='volume.bin',
            mimetype='application/octet-stream',
            metadata={'frame_count': 4},
            modality='CT',
        )
    )


def _make_dicom_resource() -> DICOMResource:
    return DICOMResource(
        **_resource_payload(
            id='dicom-resource-1',
            storage='DicomResource',
            filename='volume.dcm',
            mimetype='application/dicom',
            metadata={'frame_count': 4},
            modality='CT',
            instance_uid='1.2.3',
            series_uid='4.5.6',
            study_uid='7.8.9',
        )
    )


def _make_nifti_resource() -> NiftiResource:
    return NiftiResource(
        **_resource_payload(
            id='nifti-resource-1',
            storage='NiftiResource',
            filename='brain.nii.gz',
            mimetype='application/nifti',
            metadata={'frame_count': 4},
            modality='MR',
        )
    )


def _make_video_resource() -> VideoResource:
    return VideoResource(
        **_resource_payload(
            id='video-resource-1',
            storage='VideoResourceHandler',
            filename='cine.mp4',
            mimetype='video/mp4',
            metadata={
                'streams': [
                    {'codec_type': 'video', 'nb_frames': '6', 'width': 640, 'height': 480},
                ],
            },
            modality='US',
        )
    )


def _make_local_resource() -> LocalResource:
    return LocalResource(raw_data=b'local-resource-bytes', filename='local.bin')


def _make_local_nifti_resource() -> LocalResource:
    image = Nifti1Image(
        np.arange(27, dtype=np.int16).reshape(3, 3, 3),
        affine=np.eye(4),
    )
    return LocalResource(
        raw_data=image.to_bytes(),
        filename='brain.nii',
        storage='NiftiResource',
        mimetype='application/nifti',
        modality='MR',
    )


def _make_sliced_volume_resource() -> SlicedVolumeResource:
    r = SlicedVolumeResource(_make_local_nifti_resource(), 1, 'axial')
    _ = r.data_metainfo  # ensure cached properties are populated before pickling
    return r


def _make_sliced_video_resource() -> SlicedVideoResource:
    return SlicedVideoResource(_make_video_resource(), 1)
    


_ENTITY_FACTORIES: dict[type[object], Factory] = {
    Annotation: _make_annotation,
    BoxAnnotation: _make_box_annotation,
    Channel: _make_channel,
    DatasetInfo: _make_dataset_info,
    DICOMResource: _make_dicom_resource,
    DeployJob: _make_deploy_job,
    ImageClassification: _make_image_classification,
    ImageResource: _make_image_resource,
    ImageSegmentation: _make_image_segmentation,
    InferenceJob: _make_inference_job,
    LineAnnotation: _make_line_annotation,
    LocalResource: _make_local_resource,
    NiftiResource: _make_nifti_resource,
    Project: _make_project,
    ProjectResourceSplit: _make_project_resource_split,
    Resource: _make_resource,
    SlicedVideoResource: _make_sliced_video_resource,
    SlicedVolumeResource: _make_sliced_volume_resource,
    User: _make_user,
    VideoResource: _make_video_resource,
    VolumeResource: _make_volume_resource,
    VolumeSegmentation: _make_volume_segmentation,
}


def _discover_entity_classes() -> set[type[object]]:
    discovered: set[type[object]] = {SlicedVideoResource, SlicedVolumeResource}

    for module_info in pkgutil.walk_packages(
        entities_pkg.__path__,
        prefix=f'{entities_pkg.__name__}.',
    ):
        module = importlib.import_module(module_info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if obj in _EXCLUDED_ENTITY_CLASSES:
                continue
            if issubclass(obj, (BaseEntity, BaseEntityModel)):
                discovered.add(obj)

    return discovered


def _assert_proxy_roundtrip(
    original: SlicedVideoResource | SlicedVolumeResource,
    restored: SlicedVideoResource | SlicedVolumeResource,
) -> None:
    assert type(restored) is type(original)
    assert restored.parent_resource.model_dump(mode='json', warnings='none') == (
        original.parent_resource.model_dump(mode='json', warnings='none')
    )

    if isinstance(original, SlicedVolumeResource):
        assert restored.slice_index == original.slice_index
        assert restored.slice_axis == original.slice_axis
    else:
        assert restored.frame_index == original.frame_index


def _assert_roundtrip(original: object, restored: object) -> None:
    if isinstance(original, BaseEntityModel):
        assert type(restored) is type(original)
        assert isinstance(restored, BaseEntityModel)
        assert restored.model_dump(mode='json', warnings='none') == (
            original.model_dump(mode='json', warnings='none')
        )
        if isinstance(original, BaseEntity):
            assert getattr(restored, '_api', None) is None
        return

    if isinstance(original, (SlicedVideoResource, SlicedVolumeResource)):
        assert isinstance(restored, type(original))
        _assert_proxy_roundtrip(original, restored)
        return

    raise TypeError(f'Unsupported pickle test object: {type(original)}')


def test_entity_pickle_registry_covers_all_concrete_entities() -> None:
    discovered = _discover_entity_classes()
    missing = sorted(
        f'{entity_cls.__module__}.{entity_cls.__name__}'
        for entity_cls in discovered
        if entity_cls not in _ENTITY_FACTORIES
    )
    unexpected = sorted(
        f'{entity_cls.__module__}.{entity_cls.__name__}'
        for entity_cls in _ENTITY_FACTORIES
        if entity_cls not in discovered
    )

    assert not missing, f'Missing pickle coverage for concrete entities: {missing}'
    assert not unexpected, f'Factory registry contains unexpected classes: {unexpected}'


@pytest.mark.parametrize(
    ('entity_cls', 'factory'),
    [
        pytest.param(entity_cls, _ENTITY_FACTORIES[entity_cls], id=entity_cls.__name__)
        for entity_cls in sorted(_ENTITY_FACTORIES, key=lambda cls: cls.__name__)
    ],
)
def test_entity_roundtrips_through_pickle(entity_cls: type[object], factory: Factory) -> None:
    entity = factory()
    restored = pickle.loads(pickle.dumps(entity))

    assert type(entity) is entity_cls
    _assert_roundtrip(entity, restored)


def test_sliced_volume_resource_data_metainfo_available_after_pickle_roundtrip() -> None:
    resource = _make_sliced_volume_resource()
    restored = pickle.loads(pickle.dumps(resource))

    assert type(restored.data_metainfo) is type(resource.data_metainfo)
    assert getattr(restored.data_metainfo, 'shape', None) == getattr(resource.data_metainfo, 'shape', None)