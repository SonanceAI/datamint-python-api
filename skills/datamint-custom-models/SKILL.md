---
name: datamint-custom-models
description: Turn a user's own model into a Datamint model with the `datamint` SDK - wrap any trained model (PyTorch checkpoint, state_dict, Hugging Face, timm, foundation models, models from a custom training loop) in a `DatamintModel` adapter, test it locally, validate it, and log/register it in the Datamint MLflow registry so it can be deployed. Also covers training a plain `LightningModule` or an own Lightning loop on Datamint data with `DatamintDataModule`. Use this whenever the user already has a model, weights or training code of their own and wants to use it on Datamint, register it, or make it deployable, even if they only say "upload my model" or "use my checkpoint".
---

# Bringing your own model to Datamint

```python
from datamint import Api
api = Api()
```

A **Datamint model** is a `DatamintModel` subclass that takes Datamint resources in and returns Datamint annotations out. Once logged and registered, it can be deployed and run from the platform.

| Situation | Path |
|---|---|
| Already-trained model, any framework | Write a `DatamintModel` adapter (below) |
| Model fits `SegmentationModule` / `ClassificationModule` and the project has an annotated test split | Shortcut: `trainer.test(register_model=True)` (below) |
| Own `LightningModule` or own Lightning loop, trained on Datamint data | Train (section at the end), then write an adapter |
| New model with a Datamint trainer and a custom architecture | `datamint-training` skill |

## 1. The adapter

```python
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from datamint.entities.annotations import ImageSegmentation
from datamint.mlflow.flavors.model import DatamintModel, ModelSettings
from datamint.mlflow.flavors.task_type import TaskType

class SegmentationAdapter(DatamintModel):
    task_type = TaskType.IMAGE_SEGMENTATION

    def __init__(self, torch_model, class_names, image_size=256, threshold=0.5):
        super().__init__(torch_model=torch_model, settings=ModelSettings(need_gpu=False))
        self.class_names = class_names
        self.threshold = threshold
        self._transform = A.Compose([A.Resize(image_size, image_size), A.Normalize(), ToTensorV2()])

    def predict_default(self, model_input, **kwargs):
        device = self.inference_device
        model = self.get_pytorch_model().to(device).eval()
        results = []
        for resource in model_input:
            img = np.array(resource.fetch_file_data(auto_convert=True, use_cache=True))
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            h, w = img.shape[:2]
            x = self._transform(image=img[..., :3])['image'].unsqueeze(0).to(device)
            with torch.inference_mode():
                probs = model(x).sigmoid().squeeze(0).cpu().numpy()
            results.append([
                ImageSegmentation(name=name,
                                  segmentation_data=cv2.resize((probs[i] > self.threshold).astype(np.uint8),
                                                               (w, h), interpolation=cv2.INTER_NEAREST))
                for i, name in enumerate(self.class_names)
            ])
        return results

net = ...  # the user's model, loaded as usual (e.g. load_state_dict + eval())
adapter = SegmentationAdapter(torch_model=net, class_names=['lesion'])
```

The contract:
- `predict_default(model_input, **kwargs)` receives a `list` of resources and returns `list[list[Annotation]]`: one list per resource, in the same order. A resource with no findings gets an empty list.
- Set `task_type` as a class attribute, so the platform knows how to display predictions.
- Pass the network as `torch_model=` and get it back with `self.get_pytorch_model()`. This keeps it serializable with the model.
- Use `self.inference_device` (`'cuda'` or `'cpu'`, chosen at load time). Don't hardcode `'cuda'`.
- `ModelSettings(need_gpu=True)` when the model can't run on CPU.
- Load input with `resource.fetch_file_data(auto_convert=True, use_cache=True)`: PIL image, `pydicom.Dataset`, `nibabel` image, or video container, depending on the file.
- A Lightning `.ckpt` from a Datamint module loads with `SegmentationModule.load_from_checkpoint(path)`.
- Models already registered in MLflow can be linked instead of embedded: `super().__init__(mlflow_models_uri={'model': 'models:/OtherModel/3'})`, then `self.get_mlflow_models()['model']` inside `predict_default`.

`TaskType` values: `IMAGE_CLASSIFICATION`, `MULTILABEL_IMAGE_CLASSIFICATION`, `IMAGE_SEGMENTATION`, `INSTANCE_SEGMENTATION`, `OBJECT_DETECTION`, `VOLUME_SEGMENTATION`, `VOLUME_CLASSIFICATION`, `VIDEO_FRAME_CLASSIFICATION`, `VIDEO_SEGMENTATION`, `LANDMARK_DETECTION`, `ANOMALY_DETECTION`, `REPORT_GENERATION`.

Output annotation types (`from datamint.entities.annotations import ...`):

| Task | Return |
|---|---|
| 2D segmentation | `ImageSegmentation(name='lesion', segmentation_data=mask_uint8_HxW)` |
| 3D segmentation | `VolumeSegmentation.from_semantic_segmentation(label_volume, class_map={1: 'liver', 2: 'tumor'})` |
| Classification | `CategoryAnnotation(name='finding', value='malignant', confiability=0.93)` |
| Detection | `BoxAnnotation.from_points((x1, y1), (x2, y2), identifier='cell')` |

Annotation names must match the label names used in the Datamint project, or predictions won't line up with the project's labels.

### Prediction modes (optional)

`predict_default` handles a whole resource. To also serve one slice or frame at a time, override the matching hook. The default slice/frame/volume hooks already route through `predict_default` where possible.

```python
from datamint.mlflow.flavors.prediction_modes import PredictionMode
from datamint.mlflow.flavors.prediction_router import prediction_mode

class MyAdapter(DatamintModel):
    @prediction_mode(PredictionMode.SLICE, param_keys=('slice_index', 'axis'))
    def predict_slice(self, model_input, *, slice_index, axis='axial', **kwargs):
        ...
```

Modes: `DEFAULT`, `IMAGE`, `FRAME`, `FRAME_RANGE`, `ALL_FRAMES`, `TEMPORAL_SEQUENCE`, `SLICE`, `SLICE_RANGE`, `PRIMARY_SLICE`, `VOLUME`, `INTERACTIVE`, `FEW_SHOT`.

## 2. Test locally before logging

```python
import io
from PIL import Image
from datamint.entities.resource import LocalResource

buf = io.BytesIO()
Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)).save(buf, format='PNG')
preds = adapter.predict([LocalResource(raw_data=buf.getvalue(), filename='dummy.png')])

real = api.resources.get_list(project_name='MyProject', limit=2)
preds = adapter.predict(list(real))
```

`LocalResource(local_filepath='scan.nii.gz')` wraps a file on disk.

Validate before registering:

```python
from datamint import validate_model
report = validate_model(adapter, sample_input=list(real))   # or dataset=ImageDataset(project=...), n_samples=2
print(report)   # [v] passed, [!] warning (still deployable), [x] error (don't deploy)
```

## 3. Log and register

```python
from datamint import ImageDataset

model = api.models.log_model(
    adapter,
    project='MyProject',
    model_name='my-external-unet',
    dataset=ImageDataset(project='MyProject'),   # optional: derives annotation_specs from the project labels
)
```

`api.models.log_model` sets the project, opens an MLflow run (experiment = `model_name`), logs the adapter with the `datamint` and `medimgkit` versions pinned, registers a new version under `model_name`, and returns the registered `Model`. Each call adds a new version.

Options:
- `task_type`: defaults to the adapter's `task_type`.
- `annotation_specs`: explicit label specs, taking precedence over `dataset`.
- `supported_modes`: prediction modes the adapter implements.
- `extra_pip_requirements=['timm==1.0.9']` for packages the model needs beyond `datamint`'s own dependencies.
- `code_paths=['my_package/']` when the adapter or network class is defined in a local module (not in the running script and not an installed package), so the server can import it.

Inside an existing MLflow run (to log params and metrics alongside), use the flavor function directly. The registry name argument is `model_name`:

```python
import mlflow
import datamint.mlflow as datamint_mlflow
from datamint.mlflow.flavors import log_model

datamint_mlflow.set_project('MyProject')
mlflow.set_experiment('my-external-unet')
with mlflow.start_run(run_name='external_upload'):
    mlflow.log_params({'encoder': 'resnet34'})
    info = log_model(adapter, task_type=TaskType.IMAGE_SEGMENTATION, name='segmentation_model',
                     model_name='my-external-unet', model_config={'device': 'cpu'})
```

`model_config={'device': 'cpu'}` pins the inference device. Otherwise it's the `MLFLOW_DEFAULT_PREDICTION_DEVICE` env var, then `cuda` if available, then `cpu`.

Verify the round trip:

```python
version = api.models.get_by_name('my-external-unet').get_latest_version()   # highest version
loaded = version.load_model()
preds = loaded.predict(list(real))
```

Deployment and remote inference: `datamint-deploy` skill. A deploy with no version or alias uses the highest registered version.

## Shortcut: register with test metrics

When the model can be written as a `SegmentationModule` / `ClassificationModule` subclass and the project has an annotated test split, this evaluates it and registers it in one call, without training:

```python
from datamint.lightning import SemanticSegmentation2DTrainer
from datamint.lightning.trainers.lightning_modules import SegmentationModule

class ExternalSegModule(SegmentationModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, class_names=['lesion'], **kwargs)
        self.model = net

    def forward(self, x):
        return self.model(x)

trainer = SemanticSegmentation2DTrainer(project='MyProject', image_size=256,
                                        model=ExternalSegModule, model_name='my-external-unet')
test_metrics = trainer.test(register_model=True)
```

## Training own code on Datamint data

Both options give Datamint datasets, splits, MLflow logging and checkpointing. The result is a plain Lightning model: wrap its network in an adapter (section 1) to make it a Datamint model. Dataset item format, mask format and splits: see `datamint-training`.

Own `LightningModule` in a trainer: pass an **instance**. The module owns its loss, optimizer and steps. Masks are in `batch['masks']`, background at channel 0.

```python
trainer = SemanticSegmentation2DTrainer(project='MyProject', model=MyLightningModule(), max_epochs=20)
trainer.fit()
```

Own Lightning loop with `DatamintDataModule`:

```python
import albumentations as A
import lightning as L
from datamint import ImageDataset
from datamint.lightning import DatamintDataModule

dm = DatamintDataModule(
    ImageDataset(project='MyProject'),     # no transforms on the dataset itself
    batch_size=8, num_workers=4,
    split=True,                            # project split assignments; or {'train': 0.8, 'val': 0.1, 'test': 0.1} + split_seed=42
    train_transform=A.Compose([A.HorizontalFlip(), A.Resize(256, 256)]),
    eval_transform=A.Compose([A.Resize(256, 256)]),
)
trainer = L.Trainer(max_epochs=20)
trainer.fit(my_module, datamodule=dm)
trainer.test(my_module, datamodule=dm)
```

`split=None` uses the full dataset for every stage. `split_as_of_timestamp=` replays an exact earlier project split. `train_batch_size` / `val_batch_size` / `test_batch_size` override `batch_size` per stage.
