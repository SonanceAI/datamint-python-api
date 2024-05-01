# Datamint python API

## Installation
Install via pip: `pip install git+https://github.com/SonanceAI/datamint-python-api`.

## Setup API key
> [!NOTE]
> Not required if you don't plan to communicate with the server.

There three options to specify the API key:
1. Specify API key as an enviroment variable:
  - **command line:** `export DATAMINT_API_KEY="my_api_key"; python my_script.py` 
  - **python:** `os.environ["DATAMINT_API_KEY"] = "my_api_key"`
2. Specify API key in the API Handler constructor: TODO
3. run `datamint config`? (TODO?) and follow the instructions?


## API Usage
First, setup your api handler:
```python
from datamintapi import APIHandler

api_handler = APIHandler(api_key='my_api_key')
```
Alternatively, you can specify the API key as an environment variable and load it without specifying it in the constructor:
1. In bash, run `export DATAMINT_API_KEY="my_api_key"`
2. In your python code, just run: `api_handler = APIHandler()`

### Upload dicoms

#### Examples:

##### Upload a single dicom file
```python
batch_id = "abcd1234"
file_path = "/path/to/dicom.dcm"
dicom_id = api_handler.upload_dicom(batch_id, file_path)
print(f"Uploaded DICOM file with ID: {dicom_id}")
```

##### Upload dicom, anonymize it and add label 'pneumonia' to it
```python
batch_id = "abcd1234"
file_path = "/path/to/dicom.dcm"
dicom_id = api_handler.upload_dicom(batch_id, 
                                    file_path,
                                    anonymize=True,
                                    labels=['pneumonia'])
```

#### Upload multiple dicoms and create a new batch
```python
api_handler.create_new_batch(description='CT scans',
                             file_path='/path/to/dicom_files/',
                             mung_filename='all', # This will convert files name to 'path_to_dicom_files/1.dcm', 'path_to_dicom_files/2.dcm', etc.
                            ):
```



### Dataset
Import the custom dataset class and create an instance: 
```python 
from datamintapi import Dataset

dataset = Dataset(root='../data',
                  dataset_name='TestCTdataset',
                  version='latest',
                  api_key='my_api_key'
                )
```
and then use it in your PyTorch code as usual.

#### Test
Go to DatamintAPI directory and run `dataset.py`
```bash
cd datamintapi; python dataset.py
```

#### Examples
Here are some examples on how to use the custom dataset class:

##### Pytorch

Inheriting datamint [`Dataset`](datamintapi/dataset.py):
```python
import datamintapi
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class XrayFractureDataset(datamintapi.Dataset):
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
                              api_key='my_api_key',
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
import datamintapi
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create an instance of the datamintapi.Dataset
dataset = datamintapi.Dataset(root='data',
                              dataset_name='TestCTdataset',
                              version='latest',
                              api_key='my_api_key',
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

## Command-line Usage
### datamint-upload
Upload a single dicom file:
```bash
datamint-upload datamint-upload --path data/dicom_file.dcm
```

Upload all dicom files inside a directory and all its subdirectories, recursively:
```bash
datamint-upload --path data/dicom_files/ -r
```
(...)