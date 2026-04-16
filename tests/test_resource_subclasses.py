from datamint.api.client import Api
from datamint.entities import (
    DICOMResource,
    ImageResource,
    NiftiResource,
    Resource,
    VideoResource,
    VolumeResource,
)


def _resource_payload(**overrides) -> dict:
    payload = {
        'id': 'cd69c126-02ee-44af-8672-13d61b09eee4',
        'resource_uri': '/resources/cd69c126-02ee-44af-8672-13d61b09eee4/file',
        'storage': 'ImageResource',
        'location': 'resources/customer/cd69c126-02ee-44af-8672-13d61b09eee4',
        'upload_channel': 'Unknown',
        'filename': 'normal.png',
        'modality': 'US',
        'mimetype': 'image/png',
        'size': 444584,
        'upload_mechanism': 'api',
        'customer_id': '79113ed1-0535-4f53-9359-7fe3fa9f28a8',
        'status': 'inbox',
        'created_at': '2025-12-24T11:34:18.036Z',
        'created_by': 'datamint-dev@mail.com',
        'published': False,
        'deleted': False,
        'metadata': {'width': 693, 'height': 582},
        'source_filepath': '/tmp/normal/normal.png',
        'tags': ['ultrasound', 'split:train'],
        'user_info': {'firstname': None, 'lastname': None},
    }
    payload.update(overrides)
    return payload


class TestResourceSubclasses:
    def test_image_resource_specialization(self) -> None:
        resource = Resource(**_resource_payload())

        assert isinstance(resource, ImageResource)
        assert resource.kind == 'image'
        assert resource.get_dimensions() == (693, 582)
        assert resource.get_depth() == 1

    def test_video_resource_specialization(self) -> None:
        resource = Resource(**_resource_payload(
            storage='VideoResourceHandler',
            filename='clip.mp4',
            mimetype='video/mp4',
            metadata={
                'streams': [
                    {'codec_type': 'audio', 'nb_frames': '0'},
                    {'codec_type': 'video', 'nb_frames': '31', 'width': 1920, 'height': 1080},
                ],
            },
        ))

        assert isinstance(resource, VideoResource)
        assert resource.kind == 'video'
        assert resource.get_dimensions() == (1920, 1080)
        assert resource.frame_count == 31

    def test_dicom_resource_specialization(self) -> None:
        resource = Resource(**_resource_payload(
            storage='DicomResource',
            filename='study.dcm',
            mimetype='application/dicom',
            metadata={'frame_count': '24'},
            instance_uid='1.2.3',
            series_uid='4.5.6',
            study_uid='7.8.9',
        ))

        assert isinstance(resource, DICOMResource)
        assert resource.kind == 'dicom'
        assert resource.is_volume()
        assert resource.frame_count == 24
        assert resource.uids == {
            'instance_uid': '1.2.3',
            'series_uid': '4.5.6',
            'study_uid': '7.8.9',
        }

    def test_nifti_resource_specialization_from_filename_suffix(self) -> None:
        resource = Resource(**_resource_payload(
            storage='BinaryBlob',
            filename='brain_scan.nii.gz',
            mimetype='application/gzip',
            metadata={'frame_count': 48},
        ))

        assert isinstance(resource, NiftiResource)
        assert isinstance(resource, VolumeResource)
        assert resource.kind == 'nifti'
        assert resource.frame_count == 48
        assert resource.is_compressed is True

    def test_resources_api_initializes_specialized_entity(self) -> None:
        api = Api(
            server_url='https://api.example.com',
            api_key='test-key',
            check_connection=False,
        )

        try:
            resource = api.resources._init_entity_obj(**_resource_payload(
                storage='VideoResourceHandler',
                filename='cine.mp4',
                mimetype='video/mp4',
                metadata={'streams': [{'codec_type': 'video', 'nb_frames': '8'}]},
            ))
        finally:
            api.close()

        assert isinstance(resource, VideoResource)
        assert resource._api is api.resources