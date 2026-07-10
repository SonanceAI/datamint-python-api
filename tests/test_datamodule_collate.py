"""Tests for DatamintDataModule collate_fn support."""
from unittest.mock import MagicMock, patch

from datamint.lightning.datamodule import DatamintDataModule


def _make_dm(collate_fn=None):
    """ Helper to build a DatamintDataModule with a mock dataset. """
    mock_dataset = MagicMock()
    mock_dataset.get_collate_fn.return_value = MagicMock(name='default_collate')
    return DatamintDataModule(dataset=mock_dataset, collate_fn=collate_fn)


def test_default_collate_fn_is_none():
    dm = _make_dm()
    assert dm.collate_fn is None


def test_collate_fn_stored_when_provided():
    custom = MagicMock()
    dm = _make_dm(collate_fn=custom)
    assert dm.collate_fn is custom


def test_custom_collate_fn_forwarded_to_train_dataloader():
    custom = MagicMock()
    dm = _make_dm(collate_fn=custom)
    dm._train_dataset = dm.dataset 

    with patch('datamint.lightning.datamodule.DataLoader') as MockLoader:
        MockLoader.return_value = MagicMock()
        dm.train_dataloader()

    assert MockLoader.call_args.kwargs.get('collate_fn') is custom


def test_default_falls_back_to_dataset_collate_fn():
    dm = _make_dm()
    dm._train_dataset = dm.dataset

    with patch('datamint.lightning.datamodule.DataLoader') as MockLoader:
        MockLoader.return_value = MagicMock()
        dm.train_dataloader()

    passed = MockLoader.call_args.kwargs.get('collate_fn')
    assert passed is dm.dataset.get_collate_fn.return_value


def test_use_project_splits_forwarded_to_dataset_split():
    mock_dataset = MagicMock()
    mock_dataset.split.return_value = {"train": mock_dataset, "val": None, "test": None}
    dm = DatamintDataModule(dataset=mock_dataset, use_project_splits=True)

    dm._resolve_dataset_splits()

    assert mock_dataset.split.call_args.kwargs["use_project_splits"] is True
    assert mock_dataset.split.call_args.kwargs["use_server_splits"] is None
