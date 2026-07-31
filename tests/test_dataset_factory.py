import pytest

from datamint.dataset.factory import build_dataset


def test_build_dataset_requires_project() -> None:
    with pytest.raises(TypeError):
        build_dataset()
