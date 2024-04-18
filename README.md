# Custom PyTorch Dataset Class

## Description
This repository contains a custom PyTorch dataset class implementation that uses the Datamint API in order to loads its datasets.

## Installation
Install via pip: `pip install git+https://github.com/SonanceAI/datamint-python-api`.


## Usage
1. Specify API key as a enviroment variable. **(Optional)**: This is optional if you later specify the API in the dataset constructor, or you do not want to
download/update an existing download dataset.
    - **command line:** `export DATAMINT_API_KEY="my_api_key"; python my_script.py` 
    - **python:**
    ```python
    import os
    os.environ["DATAMINT_API_KEY"] = "my_api_key"
    ```
2. Import the custom dataset class and create an instance: 
```python 
import DatamintAPI

dataset = DatamintAPI.Dataset(root='../data',
                              dataset_name='TestCTdataset',
                              version='latest',
                              api_key='my_api_key'
                             )
```
3. Use the dataset in your PyTorch code.

## Test
Go to DatamintAPI directory and run `dataset.py`
```bash
cd DatamintAPI; python dataset.py
```

## Examples
Here are some examples on how to use the custom dataset class:

### Pytorch

Inheriting `DatamintAPI.Dataset`:
```python
import DatamintAPI
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class XrayFractureDataset(DatamintAPI.Dataset):
    def __getitem__(self, idx):
        image, dicom_metainfo, metainfo = super().__getitem__(idx)

        # Get all relevant information from the dicom_metainfo object
        patient_sex = dicom_metainfo.PatientSex

        # Get all relevant information from the metainfo object
        has_fracture = 'fracture' in metainfo['labels']
        has_fracture = torch.tensor(has_fracture, dtype=torch.int32)

        return image, patient_sex, has_fracture


# Create an instance of your custom dataset
dataset = XrayFractureDataset(root='data',
                              dataset_name='YOUR_DATASET_NAME',
                              version='latest',
                              api_key='abc123',
                              transform=ToTensor()
                              )

# Create a DataLoader to handle batching and shuffling of the dataset
dataloader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True)

for images, patients_sex, labels in dataloader:
    images = images.to(device)
    # labels will already be a tensor of shape (batch_size,) containing 0s and 1s

    # (...) do something with the batch
```

Alternative:
```python
import DatamintAPI
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create an instance of the DatamintAPI.Dataset
dataset = DatamintAPI.Dataset(root='data',
                              dataset_name='TestCTdataset',
                              version='latest',
                              api_key='abc123',
                              transform=ToTensor()
                              )

# This function tells the dataloader how to group the items in a batch


def collate_fn(batch):
    images = [item[0] for item in batch]
    dicom_metainfo = [item[1] for item in batch]
    metainfo = [item[2] for item in batch]

    return torch.stack(images), dicom_metainfo, metainfo


# Create a DataLoader to handle batching and shuffling of the dataset
dataloader = DataLoader(dataset,
                        batch_size=4,
                        collate_fn=collate_fn,
                        shuffle=True)

for images, dicom_metainfo, metainfo in dataloader:
    images = images.to(device)
    metainfo = metainfo

    # (... do something with the batch)
```
