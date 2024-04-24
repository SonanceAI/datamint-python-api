import pytest
import pydicom
from datamintapi.dicom_utils import anonymize_dicom

CLEARED_MSG = "CLEARED_BY_DATAMINT"


class TestDicomUtils:
    @pytest.fixture
    def sample_dataset1(self):
        with open('/tmp/testlog.txt', 'a') as f:
            f.write("Creating sample dataset 1\n")
        ds = pydicom.Dataset()
        ds.PatientName = "John Doe"
        ds.PatientID = "12345"
        ds.Modality = "CT"
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        return ds

    def test_anonymize_dicom(self, sample_dataset1):
        # Create a sample DICOM dataset
        ds = sample_dataset1

        # Call the anonymize_dicom function
        anonymized_ds = anonymize_dicom(ds, copy=True)

        # Check if the specified DICOM tags are cleared
        assert anonymized_ds.PatientName == CLEARED_MSG
        assert anonymized_ds.PatientID == CLEARED_MSG
        assert anonymized_ds.Modality == ds.Modality
        # Check if the SOPInstanceUID and MediaStorageSOPInstanceUID are changed
        assert anonymized_ds.SOPInstanceUID != ds.SOPInstanceUID

    def test_anonymize_dicom_with_retain_codes(self, sample_dataset1):
        # Create a sample DICOM dataset
        ds = sample_dataset1

        # Specify the retain codes
        retain_codes = [(0x0010, 0x0020)]

        # Call the anonymize_dicom function
        anonymized_ds = anonymize_dicom(ds, copy=False, retain_codes=retain_codes)

        # Check if the specified DICOM tags are retained
        assert anonymized_ds.PatientName == CLEARED_MSG
        assert anonymized_ds.PatientID == '12345'
        assert anonymized_ds.Modality == 'CT'
