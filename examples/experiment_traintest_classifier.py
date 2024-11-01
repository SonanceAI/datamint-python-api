import torch.utils
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from datamintapi.experiment.experiment import Experiment
import logging
from torchmetrics import Recall, Precision, Specificity, F1Score, Accuracy, MatthewsCorrCoef
import torchmetrics
from torchvision import transforms as T
from typing import Sequence

LOGGER = logging.getLogger(__name__)

## Set API Key ##
# run `datamint-config` in the terminal to set the API key OR set it here:
# os.environ["DATAMINT_API_KEY"] = "abc123", # uncomment this line if you have not configured the API key


# Define the network architecture:


class Classifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_labels)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x


def main():
    # Initialize the experiment. This will create a new experiment on the platform.
    exp = Experiment(name="Experiment2",
                     dataset_name='DicomDataset',
                     #  dry_run=True  # Set dry_run=True to avoid uploading the results to the platform
                     )

    ### Load dataset ###
    dataset_params = dict(
        return_frame_by_frame=True,
        transform=T.Resize((28, 28))
    )

    train_dataset = exp.get_dataset("train", **dataset_params)
    test_dataset = exp.get_dataset("test", **dataset_params)

    trainloader = train_dataset.get_dataloader(batch_size=4)
    testloader = test_dataset.get_dataloader(batch_size=4)

    ####################

    num_labels = train_dataset.num_labels

    ### Define the model, loss function, and metrics ###
    model = Classifier(num_labels=num_labels)
    metrics = [Recall(task="multilabel", num_labels=num_labels, average="macro"),
               Precision(task="multilabel", num_labels=num_labels, average="macro"),
               Specificity(task="multilabel", num_labels=num_labels, average="macro"),
               F1Score(task="multilabel", num_labels=num_labels, average="macro"),
               Accuracy(task="multilabel", num_labels=num_labels, average="macro"),
               MatthewsCorrCoef(task="multilabel", num_labels=num_labels)
               ]
    criterion = nn.BCELoss()
    ####################

    training_loop(model, criterion, trainloader, metrics)

    # Evaluate the model on the test data
    test_loop(model, criterion, testloader, metrics)
    exp.log_model(model, log_model_attributes=True)


def training_loop(model, criterion, trainloader,
                  metrics: Sequence[torchmetrics.Metric],
                  lr=0.003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 4
    with tqdm(total=epochs) as pbar:
        for e in range(epochs):
            pbar.set_description(f"Epoch {e}")
            running_loss = 0
            for batch in trainloader:
                # batch is a dictionary with keys "images", "labels"
                images = batch["image"]
                labels = batch["labels"]  # labels is a tensor of shape (batch_size, num_labels)
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
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(testloader):
            # batch is a dictionary with keys "images", "labels"
            images = batch["image"]
            labels = batch["labels"]  # labels is a tensor of shape (batch_size, num_labels)
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
    logging.getLogger().addHandler(rich.logging.RichHandler())
    main()

