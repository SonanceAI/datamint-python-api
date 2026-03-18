"""
DatamintDataModule — LightningDataModule wrapper for Datamint datasets.

Wraps any :class:`~datamint.dataset.base.DatamintBaseDataset` subclass and
provides ``train_dataloader``, ``val_dataloader``, ``test_dataloader``, and
``predict_dataloader`` for use with a Lightning :class:`~lightning.pytorch.trainer.trainer.Trainer`.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import lightning as L
from torch.utils.data import DataLoader

from datamint.dataset.base import DatamintBaseDataset

_LOGGER = logging.getLogger(__name__)


class DatamintDataModule(L.LightningDataModule):
    """A :class:`~lightning.pytorch.core.LightningDataModule` that wraps a
    :class:`~datamint.dataset.base.DatamintBaseDataset`.

    The dataset must already be fully constructed (project loaded, filters
    applied).  Splitting is delegated to :meth:`DatamintBaseDataset.split`.
    Stage-specific transforms are applied to each split after splitting.

    Args:
        dataset: A fully initialised Datamint dataset (without transforms;
            those are applied per-split via *train_transform* / *eval_transform*).
        batch_size: Default batch size for every stage.
        train_batch_size: Override batch size for training.
        val_batch_size: Override batch size for validation.
        test_batch_size: Override batch size for testing.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory in DataLoaders.
        shuffle_train: Shuffle the training dataloader.
        drop_last_train: Drop last incomplete training batch.
        split: Split ratios forwarded to :meth:`DatamintBaseDataset.split`
            (e.g. ``{'train': 0.7, 'val': 0.15, 'test': 0.15}``).
            When *None* the full dataset is used for every stage.
        split_seed: Random seed for reproducible local splits.
        use_server_splits: If *True*, use server-side ``split:*`` tags
            instead of local random splitting.
        train_transform: Albumentations transform applied **only** to the
            training split (e.g. augmentations).  Calls
            :meth:`~datamint.dataset.base.DatamintBaseDataset.set_transform`
            on the train split after :meth:`setup` resolves the splits.
        eval_transform: Albumentations transform applied to the validation
            and test splits (typically resize/normalise only, no augmentation).

    Example::

        import albumentations as A

        train_tfm = A.Compose([A.RandomHorizontalFlip(), A.Normalize()])
        eval_tfm  = A.Compose([A.Normalize()])

        dataset = ImageDataset(project='my_project', ...)
        dm = DatamintDataModule(
            dataset,
            batch_size=8,
            split={'train': 0.8, 'val': 0.1, 'test': 0.1},
            split_seed=42,
            train_transform=train_tfm,
            eval_transform=eval_tfm,
        )

        # prepare_data() / setup() fetch data and make attributes available:
        trainer = L.Trainer(...)
        trainer.fit(model, datamodule=dm)
        trainer.test(datamodule=dm)
    """

    def __init__(
        self,
        dataset: DatamintBaseDataset,
        batch_size: int = 32,
        train_batch_size: int | None = None,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        drop_last_train: bool = False,
        split: dict[str, float] | bool | None = True,
        split_seed: int | None = None,
        use_server_splits: bool | None = None,
        train_transform: Callable | None = None,
        eval_transform: Callable | None = None,
    ) -> None:
        super().__init__()
        # TODO: save the transforms as strings in the hyperparameters
        self.save_hyperparameters(ignore=["dataset", "train_transform", "eval_transform"])

        self._dataset = dataset
        self._batch_size = batch_size
        self._train_batch_size = train_batch_size or batch_size
        self._val_batch_size = val_batch_size or batch_size
        self._test_batch_size = test_batch_size or batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle_train = shuffle_train
        self._drop_last_train = drop_last_train
        if isinstance(split, bool):
            self._split = split
            self._split_cfg = None
        else:
            self._split = split is not None
            self._split_cfg = split
        self._split_seed = split_seed
        self._use_server_splits = use_server_splits
        self._train_transform = train_transform
        self._eval_transform = eval_transform

        # Populated by setup()
        self._train_dataset: DatamintBaseDataset | None = None
        self._val_dataset: DatamintBaseDataset | None = None
        self._test_dataset: DatamintBaseDataset | None = None

        # Cache the split result so setup() is idempotent
        self._splits_resolved = False

    @property
    def dataset(self) -> DatamintBaseDataset:
        """The wrapped Datamint dataset."""
        return self._dataset

    # ------------------------------------------------------------------
    # LightningDataModule lifecycle
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        self._dataset._prepare()

    def setup(self, stage: str | None = None) -> None:
        if self._splits_resolved:
            return

        if self._split or self._split_cfg is not None or self._use_server_splits:
            parts = self._dataset.split(
                seed=self._split_seed,
                use_server_splits=self._use_server_splits,
                **(self._split_cfg or {}),
            )
            self._train_dataset = parts.get("train")
            self._val_dataset = parts.get("val")
            self._test_dataset = parts.get("test")

            if stage == "fit" and self._train_dataset is None:
                raise ValueError(
                    "No 'train' split found. Make sure the split config "
                    "contains a 'train' key."
                )
        else:
            # No split config: use the full dataset for every stage.
            self._train_dataset = self._dataset
            self._val_dataset = None
            self._test_dataset = self._dataset

        # Apply stage-specific transforms after splits are resolved.
        if self._train_transform is not None and self._train_dataset is not None:
            self._train_dataset.set_transform(self._train_transform)
        if self._eval_transform is not None:
            for ds in (self._val_dataset, self._test_dataset):
                if ds is not None:
                    ds.set_transform(self._eval_transform)

        self._splits_resolved = True

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("No training dataset available. Call setup('fit') first.")
        return DataLoader(
            self._train_dataset,
            batch_size=self._train_batch_size,
            shuffle=self._shuffle_train,
            drop_last=self._drop_last_train,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._dataset.get_collate_fn(),
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._dataset.get_collate_fn(),
        )

    def test_dataloader(self) -> DataLoader:
        ds = self._test_dataset if self._test_dataset is not None else self._dataset
        return DataLoader(
            ds,
            batch_size=self._test_batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._dataset.get_collate_fn(),
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size=self._test_batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._dataset.get_collate_fn(),
        )
