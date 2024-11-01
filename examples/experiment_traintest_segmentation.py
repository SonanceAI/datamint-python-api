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
from datamintapi.experiment.experiment import Experiment
import logging
import os
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
import torchmetrics
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
import torchvision
from typing import Sequence


LOGGER = logging.getLogger(__name__)

## Set API Key ##
# run `datamint-config` in the terminal to set the API key OR set it here:
# os.environ["DATAMINT_API_KEY"] = "abc123", # uncomment this line if you have not configured the API key

def initialize_model(num_classes: int, weights):
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    model.aux_classifier = None
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, num_classes)

    return model


def main():
    # Initialize the experiment. This will create a new experiment on the platform.
    exp = Experiment(name='My First Experiment',
                     dataset_name='MyDataset',
                    #  dry_run=True  # Set dry_run=True to avoid uploading the results to the platform
                     )

    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT

    ### Load dataset ###
    dataset_params = dict(
        return_frame_by_frame=True,
        image_transform=T.Compose([T.Resize((520, 520)),
                                   T.Lambda(lambda x: torch.cat([x, x, x], dim=0)),
                                   weights.transforms()
                                   ]),
        mask_transform=T.Resize((520, 520), antialias=False, interpolation=T.InterpolationMode.NEAREST),
        # This will return the mask as a semantic segmentation tensor (#classes, H, W)
        return_as_semantic_segmentation=True,
    )

    train_dataset = exp.get_dataset("train", **dataset_params)
    test_dataset = exp.get_dataset("test", **dataset_params)

    trainloader = train_dataset.get_dataloader(batch_size=4, drop_last=True)
    testloader = test_dataset.get_dataloader(batch_size=4, drop_last=True)

    ####################

    num_segmentation_classes = train_dataset.num_segmentation_labels+1  # +1 for the background class

    ### Define the model, loss function, and metrics ###
    model = initialize_model(num_segmentation_classes, weights)
    metrics = [MeanIoU(num_classes=num_segmentation_classes),
               GeneralizedDiceScore(num_classes=num_segmentation_classes)]
    criterion = nn.CrossEntropyLoss()
    ####################

    training_loop(model, criterion, trainloader, metrics)

    # Evaluate the model on the test data
    test_loop(model, criterion, testloader, metrics)


def training_loop(model, criterion, trainloader, 
                  metrics:Sequence[torchmetrics.Metric],
                  lr=0.003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 2
    with tqdm(total=epochs) as pbar:
        for e in range(epochs):
            pbar.set_description(f"Epoch {e}")
            running_loss = 0
            for batch in trainloader:
                # batch is a dictionary with keys "images", "labels"
                images = batch["image"]
                # segmentations is a tensor of shape (batch_size, #classes, H, W)
                segmentations = batch["segmentations"]

                yhat = model(images)['out']

                loss = criterion(yhat, segmentations)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for metric in metrics:
                    metric.update(yhat > 0.5, segmentations.bool())

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


def test_loop(model, criterion, testloader, metrics: Sequence[torchmetrics.Metric]):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(testloader):
            # batch is a dictionary with keys "images", "labels"
            images = batch["image"]
            segmentations = batch["segmentations"]
            yhat = model(images)['out']
            loss = criterion(yhat, segmentations)
            for metric in metrics:
                metric.update(yhat > 0.5, segmentations.bool())
            eval_loss += loss.item()

    eval_loss /= len(testloader)

    LOGGER.info(f"Eval Loss: {eval_loss}")
    LOGGER.info("Testing metrics:")
    for metric in metrics:
        LOGGER.info(f"\t{metric.__class__.__name__}: {metric.compute()}")
        metric.reset()


if __name__ == "__main__":
    import rich.logging
    LOGGER.setLevel(logging.INFO)
    logging.getLogger().addHandler(rich.logging.RichHandler())
    main()
