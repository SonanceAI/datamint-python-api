from torch.utils.data import DataLoader
from datamint import Dataset
import lightning as L
from typing import Any
from copy import copy
import numpy as np


class DatamintDataModule(L.LightningDataModule):
    """
    LightningDataModule for Datamint datasets with train/val split.
    TODO: Add support for test and predict dataloaders.
    """

    def __init__(
        self,
        project_name: str = "./",
        batch_size: int = 32,
        image_transform=None,
        mask_transform=None,
        alb_transform=None,
        alb_train_transform=None,
        alb_val_transform=None,
        train_split: float = 0.9,
        val_split: float = 0.1,
        seed: int = 42,
        num_workers: int = 4,
        **dataset_kwargs: Any,
    ):
        super().__init__()
        self.project_name = project_name
        self.batch_size = batch_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        if alb_transform is not None and (alb_train_transform is not None or alb_val_transform is not None):
            raise ValueError("You cannot specify both `alb_transform` and `alb_train_transform`/`alb_val_transform`.")

        # Handle backward compatibility for alb_transform
        if alb_transform is not None:
            self.alb_train_transform = alb_transform
            self.alb_val_transform = alb_transform
        else:
            self.alb_train_transform = alb_train_transform
            self.alb_val_transform = alb_val_transform

        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs
        self.num_workers = num_workers

        self.dataset = None

    def prepare_data(self) -> None:
        """Download or update data if needed."""
        Dataset(
            project_name=self.project_name,
            auto_update=True,
        )

    def setup(self, stage: str = None) -> None:
        """Set up datasets and perform train/val split."""
        if self.dataset is None:
            # Create base dataset for getting indices
            self.dataset = Dataset(
                return_as_semantic_segmentation=True,
                semantic_seg_merge_strategy="union",
                return_frame_by_frame=True,
                include_unannotated=False,
                project_name=self.project_name,
                image_transform=self.image_transform,
                mask_transform=self.mask_transform,
                alb_transform=None,  # No transform for base dataset
                auto_update=False,
                **self.dataset_kwargs,
            )

            indices = list(copy(self.dataset.subset_indices))
            rs = np.random.RandomState(self.seed)
            rs.shuffle(indices)
            train_end = int(self.train_split * len(indices))
            train_idx = indices[:train_end]
            val_idx = indices[train_end:]

            self.train_dataset = copy(self.dataset).subset(train_idx)
            self.train_dataset.alb_transform = self.alb_train_transform
            self.val_dataset = copy(self.dataset).subset(val_idx)
            self.val_dataset.alb_transform = self.alb_val_transform

    def train_dataloader(self) -> DataLoader:
        return self.train_dataset.get_dataloader(batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.val_dataset.get_dataloader(batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        # Use the same dataloader as validation for testing, because we have so few samples
        return self.val_dataset.get_dataloader(batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        # Use the same dataloader as validation for testing, because we have so few samples
        return self.val_dataset.get_dataloader(batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
