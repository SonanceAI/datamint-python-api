from __future__ import annotations

import json
import logging
from pathlib import Path
import nibabel as nib
import numpy as np

_LOGGER = logging.getLogger(__name__)


class NNUNetToDatamintImporter:
    """Imports nnUNet prediction NIfTI files back into Datamint as annotations.

    After ``nnUNetTrainer.run_training()`` completes, nnUNet writes one
    ``case_NNN.nii.gz`` per input volume into a predictions directory.  This
    class matches those files back to their original Datamint resource UUIDs
    (via the sidecar case map written during export) and uploads them as
    :class:`VolumeSegmentation` annotations.

    Args:
        api: Datamint API client instance.
        dataset_dir: nnUNet dataset directory that contains
            ``datamint_case_map.json`` (e.g. ``Dataset001_CTLiver/``).
    """

    def __init__(self, api, dataset_dir: Path | str) -> None:
        self._api = api
        self.dataset_dir = Path(dataset_dir)

    def _nifti_to_segmentation(
        self,
        nifti_path: Path,
        class_map: dict[int, str],
        model_id: str | None = None,
    ):
        """Load a prediction NIfTI and build a :class:`VolumeSegmentation`.

        Args:
            nifti_path: Path to the nnUNet prediction ``.nii.gz`` file.
            class_map: ``{label_int: class_name}`` — e.g. ``{1: 'liver'}``.
            model_id: Optional MLflow model ID to tag the annotation with.

        Returns:
            A :class:`VolumeSegmentation` instance ready for upload.
        """
        from datamint.entities.annotations.volume_segmentation import VolumeSegmentation

        nifti = nib.load(str(nifti_path))
        kwargs = {}
        if model_id is not None:
            kwargs['ai_model_name'] = model_id
        return VolumeSegmentation.from_semantic_segmentation(nifti, class_map, **kwargs)

    def _load_case_map(self) -> dict[str, str]:
        """Read ``datamint_case_map.json`` and return ``{case_id: resource_uuid}``.

        Returns:
            Dict mapping nnUNet case IDs (e.g. ``'case_001'``) to Datamint
            resource UUIDs (e.g. ``'res-uuid-001'``).
        """
        path = self.dataset_dir / 'datamint_case_map.json'
        if not path.exists():
            raise FileNotFoundError(
                f"Case map not found at '{path}'. "
                "This file is written during export — re-run the export step or check "
                "that 'dataset_dir' points to the correct nnUNet dataset directory."
            )
        return json.loads(path.read_text())

    def import_predictions(
        self,
        pred_dir: Path | str,
        class_map: dict[int, str],
        mlflow_model_id: str | None = None,
    ) -> list:
        """Upload nnUNet prediction NIfTI files to Datamint as volume annotations.

        For each ``*.nii.gz`` in ``pred_dir``:

        1. Strip ``.nii.gz`` to get the case ID (e.g. ``case_001``).
        2. Look the case ID up in the case map to get the Datamint resource UUID.
        3. Build a :class:`VolumeSegmentation` from the NIfTI.
        4. Upload it via :py:meth:`api.annotations.upload_volume_segmentation`.

        Args:
            pred_dir: Directory containing nnUNet prediction ``.nii.gz`` files.
            class_map: ``{label_int: class_name}`` applied to every prediction.
            mlflow_model_id: Optional MLflow model ID tagged on each annotation.

        Returns:
            List of :class:`VolumeSegmentation` instances that were uploaded.

        Raises:
            KeyError: If a prediction filename has no matching entry in the
                case map.  The error message includes the filename and the
                available keys so the caller can diagnose the mismatch.
        """
        pred_dir = Path(pred_dir)
        case_map = self._load_case_map()
        uploaded = []

        for pred_path in sorted(pred_dir.glob('*.nii.gz')):
            # Strip both .gz and .nii suffixes to get the bare case ID.
            case_id = pred_path.name.replace('.nii.gz', '')

            if case_id not in case_map:
                raise KeyError(
                    f"Prediction file '{pred_path.name}' (case_id='{case_id}') has no "
                    f"matching entry in the case map. "
                    f"Available case IDs: {sorted(case_map.keys())}"
                )

            resource_uuid = case_map[case_id]
            seg = self._nifti_to_segmentation(pred_path, class_map, model_id=mlflow_model_id)

            self._api.annotations.upload_volume_segmentation(
                resource=resource_uuid,
                file_path=pred_path,
                name=class_map,
                ai_model_name=mlflow_model_id,
            )
            _LOGGER.info(
                "Imported prediction %s → resource %s", pred_path.name, resource_uuid
            )
            uploaded.append(seg)

        _LOGGER.info("Import complete: %d predictions uploaded.", len(uploaded))
        return uploaded
