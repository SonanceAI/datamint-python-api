from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import nibabel as nib

from datamint.mlflow.flavors.model import BaseDatamintModel
from datamint.mlflow.flavors.task_type import TaskType

_LOGGER = logging.getLogger(__name__)


class NNUNetInferenceModel(BaseDatamintModel):
    """MLflow deploy adapter that runs nnUNet inference on Datamint resources.

    Registered as an MLflow ``pyfunc`` model.  At serve time, ``load_context``
    initialises ``nnUNetPredictor`` from the bundle artifact; ``predict_volume``
    writes each resource to a temp NIfTI directory, calls
    ``predict_from_files``, and converts the output NIfTIs back to
    :class:`~datamint.entities.annotations.VolumeSegmentation` instances.

    The bundle must contain::

        nnunet_bundle/
          plans.json
          dataset.json
          fold_0/
            checkpoint_final.pth

    All three are required by
    ``nnUNetPredictor.initialize_from_trained_model_folder``.
    """

    task_type = TaskType.VOLUME_SEGMENTATION

    def __init__(
        self,
        class_map: dict[int, str],
        configuration: str = '3d_fullres',
        folds: tuple[int, ...] = (0,),
        checkpoint_name: str = 'checkpoint_final.pth',
        settings=None,
    ) -> None:
        super().__init__(settings=settings)
        self.class_map = class_map
        self.configuration = configuration
        self.folds = folds
        self.checkpoint_name = checkpoint_name
        self._predictor = None

    def load_context(self, context) -> None:
        """Initialise ``nnUNetPredictor`` from the MLflow bundle artifact.

        Called by MLflow at serve time before any ``predict`` call.  Reads
        ``context.artifacts['nnunet_bundle']`` for the bundle directory path
        and calls ``initialize_from_trained_model_folder`` with the stored
        ``folds`` and ``checkpoint_name``.

        NOTE: ``nnUNetPredictor`` is looked up through the module object (not
        via ``from ... import``) so that
        ``patch('nnunetv2.inference.predict_from_raw_data.nnUNetPredictor')``
        works correctly in tests.
        """
        super().load_context(context)
        os.environ.setdefault('nnUNet_extTrainer', str(Path(__file__).parent))
        import torch
        import nnunetv2.inference.predict_from_raw_data as _pred_mod

        bundle_path = context.artifacts['nnunet_bundle']
        predictor = _pred_mod.nnUNetPredictor(device=torch.device(self.inference_device))
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=bundle_path,
            use_folds=self.folds,
            checkpoint_name=self.checkpoint_name,
        )
        self._predictor = predictor

    def _write_resource_as_nifti(self, resource, out_dir: Path, case_index: int = 1) -> Path:
        """Write one resource as a single-channel nnUNet input NIfTI.

        nnUNet expects files named ``case_{i:03d}_0000.nii.gz`` where the
        ``_0000`` suffix is the channel index (always 0 for single-modality).

        Args:
            resource: Datamint resource to write.
            out_dir: Directory to write the NIfTI into.
            case_index: 1-based sequential case number.

        Returns:
            Path to the written file.
        """
        nifti = resource.fetch_file_data(use_cache=True, auto_convert=True)
        out_path = Path(out_dir) / f'case_{case_index:03d}_0000.nii.gz'
        # convert_format already called get_fdata() to force-load into a float64
        # cache before closing the backing file handle. Using get_fdata() here
        # returns that cache; np.asarray(nifti.dataobj) would bypass it and
        # try to re-read from the closed handle, causing ValueError.
        nifti = nib.Nifti1Image(nifti.get_fdata(), nifti.affine, nifti.header)
        nib.save(nifti, str(out_path))
        return out_path

    def _nifti_to_annotation(self, pred_path: Path):
        """Load a prediction NIfTI and build a :class:`VolumeSegmentation`.

        Args:
            pred_path: Path to a ``.nii.gz`` prediction file written by
                ``nnUNetPredictor.predict_from_files``.

        Returns:
            A :class:`VolumeSegmentation` instance with ``class_map`` set to
            ``self.class_map``.
        """
        from datamint.entities.annotations.volume_segmentation import VolumeSegmentation

        nifti = nib.load(str(pred_path))
        return VolumeSegmentation.from_semantic_segmentation(nifti, self.class_map)

    def predict_volume(self, model_input, **kwargs):
        """Run nnUNet inference on a list of Datamint resources.

        Writes each resource as a NIfTI to a temporary input directory, calls
        ``nnUNetPredictor.predict_from_files``, then converts each output NIfTI
        to a :class:`VolumeSegmentation`.  The temp directory is always cleaned
        up, even if prediction raises an exception.

        Args:
            model_input: List of Datamint resource objects.

        Returns:
            ``list[list[VolumeSegmentation]]`` — one inner list per resource,
            containing the annotations produced for that resource.
        """
        if self._predictor is None:
            raise RuntimeError(
                "Predictor is not initialised. "
                "load_context() must be called before predict_volume()."
            )

        results = []
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / 'input'
            out_dir = Path(tmp) / 'output'
            in_dir.mkdir()
            out_dir.mkdir()

            for i, resource in enumerate(model_input, start=1):
                self._write_resource_as_nifti(resource, in_dir, case_index=i)

            # predict_from_files spawns subprocesses for preprocessing which
            # crash silently in memory-constrained containers. The sequential
            # variant runs everything in the main process and is correct for
            # single-volume serving.
            self._predictor.predict_from_files_sequential(
                str(in_dir), str(out_dir),
                save_probabilities=False,
            )

            for pred_path in sorted(out_dir.glob('*.nii.gz')):
                ann = self._nifti_to_annotation(pred_path)
                results.append([ann])

        return results

    def predict_default(self, model_input, **kwargs):
        """Alias for :meth:`predict_volume`."""
        return self.predict_volume(model_input, **kwargs)
