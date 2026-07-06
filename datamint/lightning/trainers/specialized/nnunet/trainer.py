from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import filelock
import mlflow
import yaml
from rich import print as rprint

from datamint.lightning.trainers.base_trainer import BaseTrainer
from datamint.dataset.volume_dataset import VolumeDataset
from datamint.lightning.trainers.specialized.nnunet.data_export import DatamintToNNUNetExporter
from datamint.entities.annotations.annotation_spec import AnnotationSpec
from datamint.entities.annotations.types import AnnotationType

if TYPE_CHECKING:
    from datamint.entities import Project

_LOGGER = logging.getLogger(__name__)

# Registry file that maps project names → nnUNet dataset IDs.
REGISTRY_PATH = Path.home() / '.config' / 'datamintapi' / 'nnunet_dataset_ids.yaml'


class NNUNetTrainer(BaseTrainer):
    """Datamint trainer that runs nnUNet v2 as the training backend.

    Exports the project data to nnUNet Task format, runs fingerprinting,
    planning, preprocessing, and training via :class:`_DatamintNNUNetTrainer`,
    then imports predictions back as Datamint annotations.

    Args:
        project: Datamint project name or object.
        configuration: nnUNet configuration — ``'2d'``, ``'3d_fullres'``
            (default), ``'3d_lowres'``, or ``'3d_cascade_fullres'``.
        fold: Cross-validation fold index (0–4) or ``'all'``.
        dataset_id: Fixed nnUNet dataset ID (1–999).  When ``None`` the ID
            is auto-assigned from the registry.
        nnunet_work_dir: Root directory for all nnUNet I/O
            (``nnUNet_raw``, ``nnUNet_preprocessed``, ``nnUNet_results``
            are created beneath it).  Defaults to
            ``~/.cache/datamint/nnunet/``.
        continue_training: Resume from an existing checkpoint.
        channel_names: Modality index → name mapping passed to
            ``dataset.json``, e.g. ``{'0': 'CT'}``.
        num_processes_preprocessing: Workers for nnUNet's preprocessing
            step.  ``None`` lets nnUNet choose.
        max_epochs: Training epochs forwarded to nnUNetTrainer.
    """

    def __init__(
        self,
        dataset=None,
        project: 'str | Project | None' = None,
        *,
        configuration: str = '3d_fullres',
        fold: int | str = 0,
        dataset_id: int | None = None,
        nnunet_work_dir: Path | str | None = None,
        continue_training: bool = False,
        channel_names: dict[str, str] | None = None,
        num_processes_preprocessing: int | None = None,
        max_epochs: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset=dataset,
            project=project,
            max_epochs=max_epochs,
            **kwargs,
        )
        self.configuration = configuration
        self.fold = fold
        self._fixed_dataset_id = dataset_id
        self.nnunet_work_dir = (
            Path(nnunet_work_dir)
            if nnunet_work_dir is not None
            else Path.home() / '.cache' / 'datamint' / 'nnunet'
        )
        self.continue_training = continue_training
        self.channel_names = channel_names or {'0': 'CT'}
        self.num_processes_preprocessing = num_processes_preprocessing

    # ── BaseTrainer abstract methods bypassed by nnUNet ───────────────────────

    def _build_dataset(self, project: 'str | Project', **kwargs) -> VolumeDataset:
        return VolumeDataset(project=project, **kwargs)

    def _build_annotation_specs(self) -> list[AnnotationSpec]:
        from datamint.dataset import VolumeDataset
        ds = VolumeDataset(
            project=self._user_project,
            return_as_semantic_segmentation=True,
            allow_external_annotations=True,
            include_unannotated=False,
        )
        return [
            AnnotationSpec(type=AnnotationType.SEGMENTATION, scope='volume', identifier=name, required=False)
            for name in ds.seglabel_list
        ]

    def _build_model(self, *args, **kwargs):
        raise NotImplementedError(
            "NNUNetTrainer bypasses the Lightning model pipeline — "
            "_build_model is not used; nnUNet builds its own model internally."
        )

    def _train_transform(self):
        raise NotImplementedError(
            "NNUNetTrainer bypasses Lightning transforms — "
            "nnUNet handles all augmentation internally."
        )

    def _eval_transform(self):
        raise NotImplementedError(
            "NNUNetTrainer bypasses Lightning transforms — "
            "nnUNet handles all preprocessing internally."
        )

    def _loss(self):
        raise NotImplementedError(
            "NNUNetTrainer bypasses the Lightning loss — "
            "nnUNet uses its own compound loss internally."
        )

    def _metrics(self):
        raise NotImplementedError(
            "NNUNetTrainer bypasses Lightning metrics — "
            "nnUNet computes Dice internally and logs it via _DatamintNNUNetTrainer."
        )

    def _monitor_metric(self):
        raise NotImplementedError(
            "NNUNetTrainer bypasses Lightning checkpointing — "
            "nnUNet manages its own checkpoint logic."
        )

    # ── nnUNet pipeline helpers ───────────────────────────────────────────────

    def _assign_dataset_id(self) -> int:
        """Return the nnUNet dataset ID for this project, assigning one if needed.

        IDs are stored in a YAML registry at :data:`REGISTRY_PATH` so that
        concurrent training runs on different projects never share an ID.
        A :class:`filelock.FileLock` guards the read-modify-write cycle.

        If ``dataset_id`` was passed to ``__init__``, that value is returned
        directly without touching the registry.

        Returns:
            Integer dataset ID in the range 1–999.
        """
        if self._fixed_dataset_id is not None:
            return self._fixed_dataset_id

        project_name = self._project_name
        lock_path = str(REGISTRY_PATH) + '.lock'

        with filelock.FileLock(lock_path):
            if REGISTRY_PATH.exists():
                registry: dict[str, int] = yaml.safe_load(REGISTRY_PATH.read_text()) or {}
            else:
                registry = {}

            if project_name in registry:
                return registry[project_name]

            next_id = max(registry.values(), default=0) + 1
            registry[project_name] = next_id

            REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            REGISTRY_PATH.write_text(yaml.dump(registry))

        return next_id

    def _set_nnunet_env(self) -> None:
        """Set the three nnUNet environment variables required by all nnUNet internals.

        nnUNet reads ``nnUNet_raw``, ``nnUNet_preprocessed``, and
        ``nnUNet_results`` from ``os.environ`` — there is no constructor
        argument alternative.  Directories are created if they do not exist.
        """
        raw = self.nnunet_work_dir / 'raw'
        preprocessed = self.nnunet_work_dir / 'preprocessed'
        results = self.nnunet_work_dir / 'results'

        for d in (raw, preprocessed, results):
            d.mkdir(parents=True, exist_ok=True)

        os.environ['nnUNet_raw'] = str(raw)
        os.environ['nnUNet_preprocessed'] = str(preprocessed)
        os.environ['nnUNet_results'] = str(results)
        # Allow nnUNet's trainer class loader to find _DatamintNNUNetTrainer
        # during inference (initialize_from_trained_model_folder reads the
        # trainer class name from the checkpoint and looks it up by name).
        os.environ['nnUNet_extTrainer'] = str(Path(__file__).parent)
        _LOGGER.debug(
            "nnUNet env set: raw=%s preprocessed=%s results=%s extTrainer=%s",
            raw, preprocessed, results, Path(__file__).parent,
        )

    @property
    def _dataset_name(self) -> str:
        """Project name sanitised for use as an nnUNet dataset name (alphanumeric only)."""
        return self._project_name.replace('_', '').replace(' ', '')

    def _run_fingerprint_and_plan(self, dataset_id: int) -> None:
        """Run nnUNet fingerprinting and experiment planning.

        Calls ``DatasetFingerprintExtractor.run()`` then
        ``ExperimentPlanner.plan_experiment()``, and raises ``RuntimeError``
        if either expected output file is missing afterwards.

        Args:
            dataset_id: nnUNet integer dataset ID.
        """
        preprocessed_dataset_dir = (
            self.nnunet_work_dir / 'preprocessed'
            / f'Dataset{dataset_id:03d}_{self._dataset_name}'
        )
        fp_file = preprocessed_dataset_dir / 'dataset_fingerprint.json'
        plans_file = preprocessed_dataset_dir / 'nnUNetPlans.json'

        if fp_file.exists() and plans_file.exists():
            rprint(f"[green]✓[/green] Fingerprinting and planning already done for dataset — skipping.")
            return

        from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import (
            DatasetFingerprintExtractor,
        )
        from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

        rprint(f"[bold]→[/bold] Running dataset fingerprinting for dataset…")
        DatasetFingerprintExtractor(dataset_id, num_processes=8).run()

        rprint(f"[bold]→[/bold] Running experiment planning for dataset…")
        ExperimentPlanner(dataset_id, gpu_memory_target_in_gb=8.0).plan_experiment()

        if not fp_file.exists():
            raise RuntimeError(
                f"dataset_fingerprint.json was not written to {fp_file}. "
                "DatasetFingerprintExtractor may have failed silently."
            )
        if not plans_file.exists():
            raise RuntimeError(
                f"nnUNetPlans.json was not written to {plans_file}. "
                "ExperimentPlanner may have failed silently."
            )
        rprint("[green]✓[/green] Fingerprinting and planning complete.")

    def _run_preprocessing(self, dataset_id: int) -> None:
        """Run nnUNet preprocessing for the configured configuration.

        Args:
            dataset_id: nnUNet integer dataset ID.
        """
        preprocessed_dataset_dir = (
            self.nnunet_work_dir / 'preprocessed'
            / f'Dataset{dataset_id:03d}_{self._dataset_name}'
        )

        # nnUNet stores preprocessed cases in a subdirectory named by 'data_identifier'
        # from nnUNetPlans.json (e.g. 'nnUNetPlans_2d'), not the bare configuration name.
        plans_file = preprocessed_dataset_dir / 'nnUNetPlans.json'
        config_dir = preprocessed_dataset_dir / self.configuration  # fallback
        if plans_file.exists():
            import json as _json_pre
            _plans = _json_pre.loads(plans_file.read_text())
            data_id = (
                _plans.get('configurations', {})
                      .get(self.configuration, {})
                      .get('data_identifier')
            )
            if data_id:
                config_dir = preprocessed_dataset_dir / data_id

        if config_dir.exists() and any(config_dir.iterdir()):
            rprint(f"[green]✓[/green] Preprocessing already done for dataset configuration '{self.configuration}' — skipping.")
            return

        from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess

        rprint(f"[bold]→[/bold] Running preprocessing for dataset configuration '{self.configuration}'…")
        num_proc = self.num_processes_preprocessing or 4
        preprocess(
            [dataset_id],
            plans_identifier='nnUNetPlans',
            configurations=(self.configuration,),
            num_processes=(num_proc,),
        )

    def _build_nnunet_trainer(self, dataset_id: int):
        """Instantiate the nnUNet–MLflow bridge trainer.

        Reads ``nnUNetPlans.json`` and ``dataset.json`` from disk (written by
        the planning and export steps) and constructs
        :class:`_DatamintNNUNetTrainer` with those configs plus the active
        MLflow run ID.

        Args:
            dataset_id: nnUNet integer dataset ID.
            run_id: Active MLflow run ID injected into the bridge for metric
                logging.

        Returns:
            A configured :class:`_DatamintNNUNetTrainer` instance ready to
            call ``run_training()``.
        """
        import json as _json
        from datamint.lightning.trainers.specialized.nnunet._nnunet_trainer_bridge import (
            _DatamintNNUNetTrainer,
        )

        dataset_dir_name = f'Dataset{dataset_id:03d}_{self._dataset_name}'

        plans = _json.loads(
            (self.nnunet_work_dir / 'preprocessed' / dataset_dir_name / 'nnUNetPlans.json')
            .read_text()
        )
        dataset_json = _json.loads(
            (self.nnunet_work_dir / 'raw' / dataset_dir_name / 'dataset.json')
            .read_text()
        )

        # nnUNet v2.8 pops 'continue_training' from the plans dict in __init__
        # to initialise its MetaLogger (append vs. new log file).
        plans = {**plans, 'continue_training': self.continue_training}

        return _DatamintNNUNetTrainer(
            plans=plans,
            configuration=self.configuration,
            fold=self.fold,
            dataset_json=dataset_json,
        )

    def _run_prediction(self, dataset_id: int, bridge) -> Path | None:
        """Run nnUNet inference on the test split and write predictions to disk.

        Uses ``nnUNetPredictor.predict_from_files`` on the ``imagesTs/``
        directory written during export.  When ``fold='all'`` was used for
        training, all five fold checkpoints are ensembled automatically.

        Args:
            dataset_id: nnUNet integer dataset ID.
            bridge: Trained :class:`_DatamintNNUNetTrainer` instance.

        Returns:
            Path to the predictions directory, or ``None`` if there are no
            test images (i.e. no test split was exported).
        """
        import nnunetv2.inference.predict_from_raw_data as _pred_mod

        dataset_dir_name = f'Dataset{dataset_id:03d}_{self._dataset_name}'
        imagesTs_dir = self.nnunet_work_dir / 'raw' / dataset_dir_name / 'imagesTs'

        if not imagesTs_dir.exists() or not any(imagesTs_dir.glob('*.nii.gz')):
            return None

        use_folds = ('all',) if self.fold == 'all' else (self.fold,)
        rprint(f"[bold]→[/bold] Running nnUNet prediction on test split (folds={use_folds})…")

        predictor = _pred_mod.nnUNetPredictor()
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=bridge.output_folder_base,
            use_folds=use_folds,
            checkpoint_name='checkpoint_best.pth',
        )

        pred_dir = Path(bridge.output_folder_base) / 'predictions_test'
        pred_dir.mkdir(exist_ok=True)
        predictor.predict_from_files(str(imagesTs_dir), str(pred_dir), save_probabilities=False)
        rprint(f"[green]✓[/green] Predictions written to {pred_dir}")
        return pred_dir

    def _import_predictions(self, dataset_id: int, pred_dir: 'Path | None') -> None:
        """Upload nnUNet test predictions to Datamint as volume annotations.

        Reads the per-class label map from ``dataset.json`` and delegates
        uploading to :class:`NNUNetToDatamintImporter`.

        Args:
            dataset_id: nnUNet integer dataset ID.
            pred_dir: Directory containing ``*.nii.gz`` prediction files
                written by :meth:`_run_prediction`.  When ``None`` (no test
                split), this method is a no-op.
        """
        if pred_dir is None:
            return

        import json as _json
        from datamint.lightning.trainers.specialized.nnunet.data_import import (
            NNUNetToDatamintImporter,
        )

        dataset_dir_name = f'Dataset{dataset_id:03d}_{self._dataset_name}'
        dataset_dir = self.nnunet_work_dir / 'raw' / dataset_dir_name

        # Invert dataset.json labels (name→int) to class_map (int→name),
        # dropping background (label 0) since it is never uploaded as an annotation.
        dataset_json_path = dataset_dir / 'dataset.json'
        labels: dict[str, int] = _json.loads(dataset_json_path.read_text()).get('labels', {})
        class_map: dict[int, str] = {v: k for k, v in labels.items() if v != 0}

        api = self.dataset._api
        NNUNetToDatamintImporter(api, dataset_dir).import_predictions(
            pred_dir, class_map=class_map, mlflow_model_id=self.model_name or self._project_name,
            source='model_pipeline',
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def fit(self) -> dict:
        """Run the full nnUNet training pipeline.

        Steps in order:

        1. Assign a stable nnUNet dataset ID for this project.
        2. Set the three ``nnUNet_*`` environment variables.
        3. Export all project resources to nnUNet Task format on disk.
        4. Run dataset fingerprinting and experiment planning.
        5. Run preprocessing for the configured nnUNet configuration.
        6. Start an MLflow run.
        7. Instantiate ``_DatamintNNUNetTrainer`` and call ``run_training()``.
        8. Run inference on the test split (``_run_prediction``).
        9. Import predictions as Datamint annotations (``_import_predictions``).

        Returns:
            ``{'bridge': bridge}`` — the trained bridge instance.
        """
        dataset_id = self._assign_dataset_id()
        self._set_nnunet_env()

        # Resolve train / test splits from the project's server-side assignments.
        # nnUNet manages its own internal cross-validation within the train split,
        # so Datamint's 'val' split is not used for that purpose — it is treated
        # as a prediction target when no 'test' split exists.
        splits = self.dataset.split(use_project_splits=True)
        train_ds = splits.get('train')
        test_ds = splits.get('test')

        train_resources = list(train_ds.resources) if train_ds is not None else list(self.dataset.resources)
        test_resources = list(test_ds.resources) if test_ds is not None else []

        if not test_resources:
            _LOGGER.warning(
                "No 'test' split found. All data will be used for training. "
                "Predictions will not be run and test results will not be available."
            )

        split_dict: dict = {'train': train_resources}
        if test_resources:
            split_dict['test'] = test_resources

        exporter = DatamintToNNUNetExporter(
            self.nnunet_work_dir / 'raw',
            dataset_id=dataset_id,
            dataset_name=self._dataset_name,
        )
        exporter.export(split_dict, self.channel_names)

        self._run_fingerprint_and_plan(dataset_id)
        self._run_preprocessing(dataset_id)

        model_name = self.model_name or self._project_name

        with self._start_mlflow_run() as run:
            run_id = run.info.run_id
            bridge = self._build_nnunet_trainer(dataset_id)
            bridge.num_epochs = self.max_epochs
            if self.continue_training:
                from nnunetv2.run.run_training import maybe_load_checkpoint
                maybe_load_checkpoint(bridge, continue_training=True, validation_only=False)
                rprint("[bold]→[/bold] Resuming training from existing checkpoint.")
            rprint(f"[bold]→[/bold] Starting nnUNet training (epochs={self.max_epochs}, fold={self.fold}, configuration='{self.configuration}')…")
            bridge.run_training()
            pred_dir = self._run_prediction(dataset_id, bridge)
            self._import_predictions(dataset_id, pred_dir)
            self._build_deploy_adapter(dataset_id, bridge)
            mlflow.register_model(f"runs:/{run_id}/nnunet_model", model_name)
            rprint(f"[green]✓[/green] Model registered as '[bold]{model_name}[/bold]' in MLflow registry.")

        return {'bridge': bridge, 'model_name': model_name}

    def _build_deploy_adapter(self, dataset_id: int, bridge) -> None:
        """Assemble the nnUNet inference bundle and log it to MLflow as a pyfunc model.

        Copies ``nnUNetPlans.json``, ``dataset_fingerprint.json``, and
        ``checkpoint_best.pth`` into a temporary bundle directory with the
        layout expected by
        ``nnUNetPredictor.initialize_from_trained_model_folder``, then logs the
        bundle via ``mlflow.pyfunc.log_model``.

        Must be called inside an active MLflow run.

        Args:
            dataset_id: nnUNet integer dataset ID.
            bridge: Trained :class:`_DatamintNNUNetTrainer` instance — provides
                ``_best_checkpoint_path`` and ``_fold``.
        """
        import json as _json
        import shutil
        import datamint.mlflow.flavors.datamint_flavor as _datamint_flavor
        from datamint.lightning.trainers.specialized.nnunet.inference_model import (
            NNUNetInferenceModel,
        )
        
        # nnUNet writes plans.json and dataset.json to output_folder_base after training.
        # initialize_from_trained_model_folder reads both from there.
        output_folder_base = Path(bridge.output_folder_base)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / 'nnunet_bundle'
            bundle.mkdir()

            shutil.copy(output_folder_base / 'plans.json', bundle / 'plans.json')
            shutil.copy(output_folder_base / 'dataset.json', bundle / 'dataset.json')

            fold_dir = bundle / f'fold_{bridge.fold}'
            fold_dir.mkdir()

            # nnUNet uses the final checkpoint for inference.
            final_ckpt = Path(bridge.output_folder) / 'checkpoint_final.pth'
            if not final_ckpt.exists():
                raise RuntimeError(
                    f"checkpoint_final.pth not found at '{final_ckpt}'. "
                    "Training may not have completed successfully."
                )
            shutil.copy(str(final_ckpt), fold_dir / 'checkpoint_final.pth')

            labels: dict[str, int] = _json.loads(
                (output_folder_base / 'dataset.json').read_text()
            ).get('labels', {})
            class_map: dict[int, str] = {v: k for k, v in labels.items() if v != 0}

            folds = ('all',) if self.fold == 'all' else (bridge.fold,)
            adapter = NNUNetInferenceModel(
                class_map=class_map,
                configuration=self.configuration,
                folds=folds,
            )
            import importlib.metadata as _imeta
            _nnunet_ver = _imeta.version('nnunetv2')
            _datamint_flavor.log_model(
                datamint_model=adapter,
                name='nnunet_model',
                artifacts={'nnunet_bundle': str(bundle)},
                extra_pip_requirements=[f'nnunetv2=={_nnunet_ver}'],
                annotation_specs=self._build_annotation_specs(),
            )
