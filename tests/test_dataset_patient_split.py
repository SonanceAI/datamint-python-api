"""Tests for patient-wise dataset splitting: group_by_patient() and split(by_patient=True)."""
import pytest

from datamint.dataset.base import DatamintBaseDataset


# ---------------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------------

class _TestDataset(DatamintBaseDataset):
    """Minimal concrete subclass that bypasses API calls entirely."""

    def _reinit_api(self) -> None:
        pass  # no-op: prevents copy.copy → __setstate__ from touching the network

    def _get_raw_item(self, index: int) -> dict:
        return {'image': None, 'metainfo': {}, 'annotations': []}

    def apply_alb_transform(self, img, targets):
        return {'image': img, **targets}


class _MockResource:
    def __init__(self, resource_id: str, patient_id: str | None, metadata: dict | None = None):
        self.id = resource_id
        self.patient_id = patient_id
        self.metadata = metadata or {}

    def get_patient_id(self) -> str | None:
        if self.patient_id is not None:
            return self.patient_id
        return self.metadata.get('PatientID')


def _make_resource(resource_id: str, patient_id: str | None, metadata: dict | None = None):
    return _MockResource(resource_id, patient_id, metadata)


def _make_dataset(patient_ids: list[str | None]) -> _TestDataset:
    """Create a prepared dataset with one resource per entry in patient_ids."""
    ds = object.__new__(_TestDataset)
    resources = [_make_resource(f'res-{i}', pid) for i, pid in enumerate(patient_ids)]
    ds.__dict__.update({
        'resources': resources,
        'resource_annotations': [[] for _ in resources],
        'project': None,
        'split_name': None,
        'split_source': None,
        'split_as_of_timestamp': None,
        '_is_prepared': True,
        '_DatamintBaseDataset__api': None,
        '_server_url': None,
        '_api_key': None,
        '_auto_update': False,
        '_logged_uint16_conversion': False,
    })
    return ds


# ---------------------------------------------------------------------------
# group_by_patient()
# ---------------------------------------------------------------------------

class TestGroupByPatient:
    def test_basic_grouping(self):
        "Test that resources are grouped by patient_id, and that the correct number of resources end up in each group. " 
        # pat-A has 2 resources, pat-B has 3, pat-C has 1
        ds = _make_dataset(['pat-A', 'pat-A', 'pat-B', 'pat-B', 'pat-B', 'pat-C'])
        groups = ds.group_by_patient()

        assert set(groups.keys()) == {'pat-A', 'pat-B', 'pat-C'}
        assert len(groups['pat-A']) == 2
        assert len(groups['pat-B']) == 3
        assert len(groups['pat-C']) == 1

    def test_resources_correctly_assigned(self):
        "Test that the correct resources end up in each patient group, based on their patient_id."
        ds = _make_dataset(['pat-A', 'pat-B', 'pat-A'])
        groups = ds.group_by_patient()

        a_ids = {r.id for r in groups['pat-A'].resources}
        b_ids = {r.id for r in groups['pat-B'].resources}
        assert a_ids == {'res-0', 'res-2'}
        assert b_ids == {'res-1'}

    def test_total_resources_preserved(self):
        "Test that all resources in the original dataset are accounted for in the patient groups, with no duplicates or omissions."
        patient_ids = ['A', 'A', 'B', 'C', 'C', 'C']
        ds = _make_dataset(patient_ids)
        groups = ds.group_by_patient()

        total = sum(len(g) for g in groups.values())
        assert total == len(ds)

    def test_metadata_fallback(self):
        "Test that if a resource has no patient_id, the method falls back to looking for a PatientID in the metadata. "
        resources = [
            _make_resource('res-0', None, metadata={'PatientID': 'pat-meta'}),
            _make_resource('res-1', 'pat-direct', metadata={}),
        ]
        ds = object.__new__(_TestDataset)
        ds.__dict__.update({
            'resources': resources,
            'resource_annotations': [[], []],
            'project': None, 'split_name': None, 'split_source': None,
            'split_as_of_timestamp': None, '_is_prepared': True,
            '_DatamintBaseDataset__api': None, '_server_url': None,
            '_api_key': None, '_auto_update': False, '_logged_uint16_conversion': False,
        })
        groups = ds.group_by_patient()

        assert 'pat-meta' in groups
        assert 'pat-direct' in groups
        assert len(groups) == 2

    def test_none_strategy_individual(self):
        "Test that when none_patient_id_strategy='individual', resources with None patient_id are grouped separately, and do not interfere with real patient groups."
        ds = _make_dataset([None, None, 'pat-A'])
        groups = ds.group_by_patient(none_patient_id_strategy='individual')

        # 2 individual None groups + 1 real patient
        assert len(groups) == 3
        assert 'pat-A' in groups
        assert None not in groups

    def test_none_strategy_group(self):
        "Test that when none_patient_id_strategy='group', all resources with None patient_id are grouped together under a single None key, and do not interfere with real patient groups."
        ds = _make_dataset([None, None, 'pat-A'])
        groups = ds.group_by_patient(none_patient_id_strategy='group')

        assert None in groups
        assert len(groups[None]) == 2
        assert 'pat-A' in groups

    def test_none_strategy_skip(self):
        "Test that when none_patient_id_strategy='skip', resources with None patient_id are excluded from the groups, and do not interfere with real patient groups."
        ds = _make_dataset([None, 'pat-A', None])
        groups = ds.group_by_patient(none_patient_id_strategy='skip')

        assert None not in groups
        assert set(groups.keys()) == {'pat-A'}
        assert len(groups['pat-A']) == 1

    def test_none_strategy_error(self):
        "Test that when none_patient_id_strategy='error', if any resource has a None patient_id, a ValueError is raised indicating that no patient_id was found."
        ds = _make_dataset(['pat-A', None])
        with pytest.raises(ValueError, match='no patient_id'):
            ds.group_by_patient(none_patient_id_strategy='error')

    def test_invalid_strategy_raises(self):
        "Test that if an invalid value is passed for none_patient_id_strategy, a ValueError is raised indicating that the strategy must be one of the allowed options."
        ds = _make_dataset(['pat-A'])
        with pytest.raises(ValueError, match='must be one of'):
            ds.group_by_patient(none_patient_id_strategy='invalid')  # type: ignore[arg-type]

    def test_all_same_patient(self):
        "Test that if all resources have the same patient_id, they are all grouped together under that patient_id key."
        ds = _make_dataset(['pat-A', 'pat-A', 'pat-A'])
        groups = ds.group_by_patient()
        assert list(groups.keys()) == ['pat-A']
        assert len(groups['pat-A']) == 3


# ---------------------------------------------------------------------------
# split(by_patient=True)
# ---------------------------------------------------------------------------

class TestSplitByPatient:
    def test_no_patient_leakage(self):
        "Test that when splitting by patient, no patient appears in more than one split, even if they have multiple resources. "
        patient_ids = [pid for pid in 'ABCDEF' for _ in range(2)]
        ds = _make_dataset(patient_ids)
        parts = ds.split(train=0.7, test=0.3, by_patient=True, seed=42)

        train_pids = {r.get_patient_id() for r in parts['train'].resources}
        test_pids = {r.get_patient_id() for r in parts['test'].resources}
        assert train_pids & test_pids == set()

    def test_all_resources_accounted_for(self):
        "Test that when splitting by patient, all resources from the original dataset are included in one of the splits, with no duplicates or omissions, even if multiple resources belong to the same patient."
        ds = _make_dataset(['A', 'A', 'B', 'B', 'B', 'C'])
        parts = ds.split(train=0.7, test=0.3, by_patient=True, seed=0)

        total = sum(len(p) for p in parts.values())
        assert total == len(ds)

    def test_split_metadata_set(self):
        "Test that when splitting by patient, the resulting split datasets have their split_name and split_source attributes set correctly to indicate the type of split performed."
        ds = _make_dataset(['A', 'B', 'C', 'D'])
        parts = ds.split(train=0.5, test=0.5, by_patient=True, seed=0)

        assert parts['train'].split_name == 'train'
        assert parts['test'].split_name == 'test'
        assert parts['train'].split_source == 'local_by_patient'
        assert parts['test'].split_source == 'local_by_patient'

    def test_seed_reproducible(self):
        "Test that when splitting by patient with a specific random seed, the same patients are assigned to the same splits across multiple runs, even if patients have multiple resources. "
        ds = _make_dataset(['A', 'A', 'B', 'B', 'C', 'C'])
        parts1 = ds.split(train=0.7, test=0.3, by_patient=True, seed=7)
        parts2 = ds.split(train=0.7, test=0.3, by_patient=True, seed=7)

        ids1 = [r.id for r in parts1['train'].resources]
        ids2 = [r.id for r in parts2['train'].resources]
        assert ids1 == ids2

    def test_different_seeds_produce_different_splits(self):
        "Test that when splitting by patient with different random seeds, different patients are assigned to the splits across runs, even if patients have multiple resources. "
        ds = _make_dataset([str(i) for i in range(10)])
        parts1 = ds.split(train=0.5, test=0.5, by_patient=True, seed=1)
        parts2 = ds.split(train=0.5, test=0.5, by_patient=True, seed=99)

        ids1 = {r.id for r in parts1['train'].resources}
        ids2 = {r.id for r in parts2['train'].resources}
        assert ids1 != ids2

    def test_three_way_split_no_leakage(self):
        "Test that when performing a three-way split by patient, no patient appears in more than one split, even if they have multiple resources. "
        ds = _make_dataset([str(i) for i in range(9)])
        parts = ds.split(train=0.7, val=0.15, test=0.15, by_patient=True, seed=42)

        assert set(parts.keys()) == {'train', 'val', 'test'}
        train_pids = {r.get_patient_id() for r in parts['train'].resources}
        val_pids = {r.get_patient_id() for r in parts['val'].resources}
        test_pids = {r.get_patient_id() for r in parts['test'].resources}
        assert not (train_pids & val_pids)
        assert not (train_pids & test_pids)
        assert not (val_pids & test_pids)

    def test_patients_not_split_across_boundaries(self):
        "Test that when splitting by patient, if a patient has multiple resources, all of their resources end up in the same split, and that no patient appears in more than one split. "
        patient_ids = ['big'] * 5 + ['A', 'B', 'C', 'D', 'E']
        ds = _make_dataset(patient_ids)
        parts = ds.split(train=0.5, test=0.5, by_patient=True, seed=0)

        all_big_splits = set()
        for split_name, split_ds in parts.items():
            for r in split_ds.resources:
                if r.patient_id == 'big':
                    all_big_splits.add(split_name)

        assert len(all_big_splits) == 1, "Patient 'big' appears in more than one split"

    def test_mutual_exclusion_with_project_splits(self):
        "Test that when splitting by patient, if use_project_splits=True is also passed, a ValueError is raised indicating that the two options cannot be combined."
        ds = _make_dataset(['A', 'B'])
        with pytest.raises(ValueError, match='cannot be combined'):
            ds.split(train=0.7, test=0.3, by_patient=True, use_project_splits=True)

    def test_mutual_exclusion_with_server_splits(self):
        "Test that when splitting by patient, if use_server_splits=True is also passed, a ValueError is raised indicating that the two options cannot be combined."
        ds = _make_dataset(['A', 'B'])
        with pytest.raises(ValueError, match='cannot be combined'):
            ds.split(train=0.7, test=0.3, by_patient=True, use_server_splits=True)

    def test_requires_ratio_kwargs(self):
        "Test that when splitting by patient, if no ratio kwargs (train, val, test) are provided, a ValueError is raised indicating that at least one ratio must be specified."
        ds = _make_dataset(['A', 'B'])
        with pytest.raises(ValueError, match='requires ratio'):
            ds.split(by_patient=True)
