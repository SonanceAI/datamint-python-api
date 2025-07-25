"""
This example demonstrates how to fine-tune a segmentation model on the platform.

The example uses the `deeplabv3_mobilenet_v3_large` model from torchvision, which is pre-trained on the COCO dataset.
The model is fine-tuned on a custom dataset using the CrossEntropyLoss loss function
and evaluated using the MeanIoU and GeneralizedDiceScore metrics.

The example demonstrates the following steps:
1. Initialize the experiment
2. Load the dataset
3. Define the model, loss function, and metrics
4. Train the model
5. Evaluate the model on the test data
"""

import torch.utils
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
from datamint import Experiment
import logging
import os
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
import torchmetrics
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchmetrics import Recall, Precision, Specificity, F1Score, Accuracy, MatthewsCorrCoef
import torchvision
from typing import Sequence
from torchvision.transforms import v2

LOGGER = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

## Set API Key ##
# run `datamint-config` in the terminal to set the API key OR set it here:
# os.environ["DATAMINT_API_KEY"] = "abc123", # uncomment this line if you have not configured the API key


class ClsMetricForSegmentation(torchmetrics.Metric):
    """
    This class is used to convert the segmentation output to a classification output.
    The segmentation output is a tensor of shape (batch_size, num_classes, H, W).
    We convert it to a tensor of shape (batch_size, num_classes) by taking the maximum value along the height and width
    """

    def __init__(self, num_labels: int, metrics, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.metrics = metrics

    def update(self, yhat, y):
        yhat_cls = yhat.amax(dim=(2, 3)).float()  # yhat_cls.shape = (batch_size, num_classes)
        y_cls = y.amax(dim=(2, 3)).float()  # y_cls.shape = (batch_size, num_classes)
        for metric in self.metrics:
            metric.update(yhat_cls, y_cls)

    def compute(self):
        return {metric.__class__.__name__: metric.compute().item() for metric in self.metrics}

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def to(self, device):
        for metric in self.metrics:
            metric.to(device)


def initialize_model(num_classes: int, weights):
    # Load the pre-trained model.
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    model.aux_classifier = None
    # Freezing the weights of the model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head with a new one that has the correct number of output classes.
    # This is specific to the deeplabv3_mobilenet_v3_large model.
    # For other models, you may need to replace a different part of the model.
    model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, num_classes)

    return model


def main():
    # Initialize the experiment. Creates a new experiment on the platform.
    exp = Experiment(name='experiment16',
                     project_name='testproject',
                     allow_existing=True,  # If an experiment with the same name exists, allow_existing=True returns the existing experiment
                     dry_run=True  # Set dry_run=True to avoid uploading the results to the platform
                     )

    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT

    ### Load dataset ###
    dataset_params = dict(
        return_frame_by_frame=True,
        image_transform=v2.Compose([v2.Resize((520, 520)),
                                   v2.RGB(),
                                   weights.transforms()
                                    ]),
        mask_transform=v2.Resize((520, 520), antialias=False, interpolation=v2.InterpolationMode.NEAREST),
        return_segmentations=True,
        # This will return the mask as a semantic segmentation tensor (#classes, H, W)
        return_as_semantic_segmentation=True,
        semantic_seg_merge_strategy='union',
    )

    # Load the train and test datasets. This returns a subclass of PyTorch dataset object.
    train_dataset = exp.get_dataset("train", **dataset_params)
    test_dataset = exp.get_dataset("test", **dataset_params)

    # Create dataloaders for the train and test datasets.
    # This method has the convinience of automatically dealing with collate_fn.
    trainloader = train_dataset.get_dataloader(batch_size=2, drop_last=True)
    testloader = test_dataset.get_dataloader(batch_size=2, drop_last=True)

    ####################

    num_segmentation_classes = len(train_dataset.segmentation_labels_set)+1  # +1 for the background class

    ### Define the model, loss function, and metrics ###
    model = initialize_model(num_segmentation_classes, weights)
    metrics = [MeanIoU(num_classes=num_segmentation_classes),
               GeneralizedDiceScore(num_classes=num_segmentation_classes)]

    cls_metrics_params = dict(
        task="multilabel" if num_segmentation_classes > 1 else "binary",
        num_labels=num_segmentation_classes,
        average="macro"
    )

    # These metrics will be used when converting the segmentation output to classification output.
    cls_metrics = [Recall(**cls_metrics_params),
                   Precision(**cls_metrics_params),
                   Specificity(**cls_metrics_params),
                   F1Score(**cls_metrics_params),
                   Accuracy(**cls_metrics_params),
                   #    MatthewsCorrCoef(task="multilabel", num_labels=num_labels)
                   ]
    metrics.append(ClsMetricForSegmentation(num_segmentation_classes, cls_metrics))
    criterion = nn.CrossEntropyLoss()
    ####################

    training_loop(model, criterion, trainloader, metrics)
    exp.log_model(model)

    # Evaluate the model on the test data
    test_loop(model, criterion, testloader, metrics, exp)


def training_loop(model, criterion, trainloader,
                  metrics: Sequence[torchmetrics.Metric],
                  lr=0.003):
    # To device
    model.to(DEVICE)
    model.train()
    for metric in metrics:
        metric.to(DEVICE)
    criterion.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 2
    with tqdm(total=epochs) as pbar:
        for e in range(epochs):
            pbar.set_description(f"Epoch {e}")
            running_loss = 0
            for batch in trainloader:
                # batch is a dictionary with keys "images", "labels"
                images = batch["image"].to(DEVICE)
                # segmentations is a tensor of shape (batch_size, #classes, H, W)
                segmentations = batch["segmentations"].to(DEVICE)

                yhat = model(images)['out']  # yhat.shape = (batch_size, #classes, H, W)

                loss = criterion(yhat, segmentations)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for metric in metrics:
                    metric.update(yhat > 0.0, segmentations.bool())

                running_loss += loss.item()
            epoch_loss = running_loss / len(trainloader)
            LOGGER.info(f"Training loss: {epoch_loss}")
            pbar.set_postfix(loss=epoch_loss)
            pbar.update(1)
        LOGGER.info("Training metrics:")
        for metric in metrics:
            LOGGER.info(f"\t{metric.__class__.__name__}: {metric.compute()}")
            metric.reset()

    LOGGER.info("Finished training")


def test_loop(model, criterion,
              testloader: torch.utils.data.DataLoader,
              metrics: Sequence[torchmetrics.Metric],
              exp: Experiment):
    # To device
    model.to(DEVICE)
    model.eval()
    for metric in metrics:
        metric.to(DEVICE)
    criterion.to(DEVICE)

    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(testloader):
            # batch is a dictionary with keys "images", "labels"
            images = batch["image"].to(DEVICE)
            # segmentations is a tensor of shape (batch_size, #classes, H, W)
            segmentations = batch["segmentations"].to(DEVICE)
            # yhat.shape = (batch_size, #classes, H, W). Not normalized (-inf, +inf)
            yhat = model(images)['out']
            # remove background
            loss = criterion(yhat, segmentations)
            for metric in metrics:
                metric.update(yhat > 0.0, segmentations.bool())
            eval_loss += loss.item()

            yhat = yhat[:, 1:]
            yhat = torch.sigmoid(yhat)
            exp.log_semantic_seg_predictions(yhat.cpu().numpy(),
                                             resource_ids=[b['id'] for b in batch['metainfo']],
                                             label_names=testloader.dataset.segmentation_labels_set,
                                             frame_idxs=[b['frame_index'] for b in batch['metainfo']],
                                             threshold=0.5
                                             )

    eval_loss /= len(testloader)

    LOGGER.info(f"Eval Loss: {eval_loss}")
    LOGGER.info("Testing metrics:")
    for metric in metrics:
        LOGGER.info(f"\t{metric.__class__.__name__}: {metric.compute()}")
        metric.reset()


if __name__ == "__main__":
    import rich.logging
    LOGGER.setLevel(logging.INFO)
    logging.getLogger('datamint').setLevel(logging.INFO)
    logging.getLogger().addHandler(rich.logging.RichHandler())
    main()
