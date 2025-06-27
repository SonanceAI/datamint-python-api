import torch.utils
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from datamint import Experiment
import logging
from torchmetrics import Recall, Precision, Specificity, F1Score, Accuracy, MatthewsCorrCoef
import torchmetrics
from torchvision.transforms import v2 as T
from torchvision.models import resnet18, ResNet18_Weights
from typing import Sequence

LOGGER = logging.getLogger(__name__)

NUM_EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

## Set API Key ##
# run `datamint-config` in the terminal to set the API key OR set it here:
# os.environ["DATAMINT_API_KEY"] = "abc123", # uncomment this line if you have not configured the API key


# Define the network architecture:

class MyModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Freeze all layers except the last one
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_labels)

    def forward(self, x):
        return F.sigmoid(self.resnet(x))


def main():
    # Initialize the experiment. Creates a new experiment on the platform.
    exp = Experiment(name="Test Experiment1",
                     project_name='testproject',
                     allow_existing=True,  # If an experiment with the same name exists, allow_existing=True returns the existing experiment
                      dry_run=True  # Set True to avoid uploading the results to the platform
                     )

    ### Load dataset ###
    dataset_params = dict(
        return_frame_by_frame=True,
        image_transform=T.Compose([T.RGB(),  # Resnet18 expects 3 channels
                                   ResNet18_Weights.DEFAULT.transforms()]
                                  ),
        return_segmentations=False,  # We just want frame labels for classification
    )

    train_dataset = exp.get_dataset("train", **dataset_params)
    test_dataset = exp.get_dataset("test", **dataset_params)

    trainloader = train_dataset.get_dataloader(batch_size=8)
    testloader = test_dataset.get_dataloader(batch_size=8)

    ####################

    num_labels = len(train_dataset.frame_labels_set)
    if num_labels == 0:
        raise ValueError("The dataset does not have any frame labels!")
    print(f"Number of labels: {num_labels}")
    task = "multilabel" if num_labels > 1 else "binary"

    cls_metrics_params = dict(
        task=task,
        num_labels=num_labels,
        average="macro"
    )

    ### Define the model, loss function, and metrics ###
    model = MyModel(num_labels=num_labels)
    metrics = [Recall(**cls_metrics_params),
               Precision(**cls_metrics_params),
               Specificity(**cls_metrics_params),
               F1Score(**cls_metrics_params),
               Accuracy(**cls_metrics_params),
               MatthewsCorrCoef(task=task, num_labels=num_labels)
               ]
    criterion = nn.BCELoss()
    ####################

    training_loop(model, criterion, trainloader, metrics)

    # Evaluate the model on the test data
    test_loop(model, criterion, testloader, metrics)


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
    with tqdm(total=NUM_EPOCHS) as pbar:
        for e in range(NUM_EPOCHS):
            pbar.set_description(f"Epoch {e}")
            running_loss = 0
            for batch in trainloader:
                # batch is a dictionary with keys "images", "labels"
                images = batch["image"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)  # labels is a tensor of shape (batch_size, num_labels)
                yhat = model(images)
                loss = criterion(yhat, labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for metric in metrics:
                    metric.update(yhat, labels)

                running_loss += loss.item()
            epoch_loss = running_loss/len(trainloader)
            LOGGER.info(f"Training loss: {epoch_loss}")
            pbar.set_postfix(loss=epoch_loss)
            pbar.update(1)
        LOGGER.info("Training metrics:")
        for metric in metrics:
            LOGGER.info(f"\t{metric.__class__.__name__}: {metric.compute()}")
            metric.reset()

    LOGGER.info("Finished training")


def test_loop(model, criterion, testloader,
              metrics: Sequence[torchmetrics.Metric]):
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
            labels = batch["labels"].to(DEVICE)  # labels is a tensor of shape (batch_size, num_labels)

            pred = model(images)
            loss = criterion(pred, labels.float())
            for metric in metrics:
                metric.update(pred, labels)
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
    logging.getLogger('datamint').setLevel(logging.DEBUG)
    logging.getLogger().addHandler(rich.logging.RichHandler())
    main()
