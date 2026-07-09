from unittest.mock import MagicMock, patch

import pytest

from datamint.dataset.factory import build_dataset


def test_build_dataset_project_name_deprecated_alias() -> None:
    fake_api = MagicMock()
    fake_api.resources.get_list.return_value = []

    with patch("datamint.Api", return_value=fake_api):
        with pytest.warns(DeprecationWarning, match="project_name"):
            with pytest.raises(ValueError, match="MyProject"):
                build_dataset(project_name="MyProject")

    fake_api.resources.get_list.assert_called_once_with(project_name="MyProject", limit=5)


def test_build_dataset_requires_project() -> None:
    with pytest.raises(TypeError):
        build_dataset()
