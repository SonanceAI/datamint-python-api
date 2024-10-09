import torch.utils
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from datamintapi.experiment._patcher import initialize_automatic_logging
from datamintapi.experiment.experiment import Experiment
import logging
import os
from torchmetrics import Recall, Precision
import torchmetrics

LOGGER = logging.getLogger(__name__)

initialize_automatic_logging()

os.environ["DATAMINT_API_KEY"] = "abc123"

# Define the network architecture:
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


def main():
    model = Classifier()
    criterion = nn.NLLLoss()
    exp = Experiment(name="test_experiment",
                     dataset_name='dataset1',
                     api_key='abc123')

    trainloader = exp.get_dataset("train").get_dataloader(batch_size=1)
    testloader = exp.get_dataset("test").get_dataloader(batch_size=1)

    training_loop(model, criterion, trainloader)
    exp.log_model(model)

    # Evaluate the model on the test data
    metrics = [Recall(task="multiclass", num_classes=10, average="macro"),
               Precision(task="multiclass", num_classes=10, average="macro")]
    test_loop(model, criterion, testloader, metrics)

    exp.finish()


def training_loop(model, criterion, trainloader, lr=0.003):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 2
    with tqdm(total=epochs) as pbar:
        for e in range(epochs):
            pbar.set_description(f"Epoch {e}")
            running_loss = 0
            for images, labels in trainloader:
                logps = model(images)
                loss = criterion(logps, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            epoch_loss = running_loss/len(trainloader)
            LOGGER.info(f"Training loss: {epoch_loss}")
            pbar.set_postfix(loss=epoch_loss)
            pbar.update(1)

    LOGGER.info("Finished training")


def test_loop(model, criterion, testloader, metrics: torchmetrics.Metric):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            logps = model(images)
            pred = torch.exp(logps)
            loss = criterion(logps, labels)
            for metric in metrics:
                metric.update(pred, labels)
            eval_loss += loss.item()

    eval_loss /= len(testloader)

    LOGGER.info(f"Eval Loss: {eval_loss}")
    for metric in metrics:
        LOGGER.info(f"{metric.__class__.__name__}: {metric.compute()}")


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    main()
