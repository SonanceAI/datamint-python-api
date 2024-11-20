import torch.utils
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from datamintapi import Experiment
import logging
from torchmetrics import Recall, Precision, Specificity, F1Score, Accuracy, MatthewsCorrCoef
import torchmetrics
from torchvision import transforms as T
from typing import Sequence, Dict

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
    exp = Experiment(name="Experiment10",
                    #   dataset_name='project_test_dataset',
                    dataset_name='project_project3_dataset',
                    #  project_name='project3',
                    #  dry_run=True  # Set dry_run=True to avoid uploading the results to the platform
                     )

    ### Load dataset ###
    dataset_params = dict(
        return_frame_by_frame=True,
        image_transform=T.Resize((28, 28))
    )

    train_dataset = exp.get_dataset("train", **dataset_params)
    test_dataset = exp.get_dataset("test", **dataset_params)

    trainloader = train_dataset.get_dataloader(batch_size=4)
    testloader = test_dataset.get_dataloader(batch_size=4)

    ####################

    num_labels = train_dataset.num_labels
    print(f"Number of labels: {num_labels}")
    task = "multilabel" if num_labels > 1 else "binary"

    cls_metrics_params = dict(
        task=task,
        num_labels=num_labels,
        average="macro"
    )

    ### Define the model, loss function, and metrics ###
    model = Classifier(num_labels=num_labels)
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
    results = test_loop(model, criterion, testloader, metrics)

    exp.log_classification_predictions(predictions_conf=results['predictions'],
                                       label_names=test_dataset.labels_set,
                                       resource_ids=results['resource_ids'],
                                       dataset_split="test",
                                       frame_idxs=results['frame_idxs']
                                       )


def training_loop(model, criterion, trainloader,
                  metrics: Sequence[torchmetrics.Metric],
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
              metrics: Sequence[torchmetrics.Metric]) -> Dict:
    model.eval()
    eval_loss = 0

    predictions = []
    resourse_ids = []
    frame_idxs = []

    with torch.no_grad():
        for batch in tqdm(testloader):
            # batch is a dictionary with keys "images", "labels"
            images = batch["image"]
            labels = batch["labels"]  # labels is a tensor of shape (batch_size, num_labels)
            resourse_ids_i = [b['id'] for b in batch['metainfo']]
            frame_idxs_i = [b['frame_index'] for b in batch['metainfo']]

            pred = model(images)
            loss = criterion(pred, labels.float())
            for metric in metrics:
                metric.update(pred, labels)
            eval_loss += loss.item()

            # For logging
            predictions.append(pred)
            resourse_ids.extend(resourse_ids_i)
            frame_idxs.extend(frame_idxs_i)

    eval_loss /= len(testloader)

    LOGGER.info(f"Eval Loss: {eval_loss}")
    LOGGER.info("Testing metrics:")
    for metric in metrics:
        LOGGER.info(f"\t{metric.__class__.__name__}: {metric.compute()}")
        metric.reset()

    return {'predictions': torch.cat(predictions),
            'resource_ids': resourse_ids,
            'frame_idxs': frame_idxs}


if __name__ == "__main__":
    import rich.logging
    LOGGER.setLevel(logging.INFO)
    logging.getLogger('datamintapi').setLevel(logging.INFO)
    logging.getLogger().addHandler(rich.logging.RichHandler())
    main()
