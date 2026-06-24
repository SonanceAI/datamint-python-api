import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.rule import Rule

console = Console()

# ---------------------------------------------------------------------------
# Script templates — detection task
# ---------------------------------------------------------------------------

_README = """\
# __PROJECT_NAME__

Object detection project using Datamint + YOLOX.

## Getting started

Run the scripts in order:

1. `01_upload_data.py` - Upload images and create the Datamint project
2. `02_explore.py` - Inspect uploaded resources and preview annotated examples
3. `03_dataset.py` - Preview the dataset, class map, and train/val/test splits
4. `04_train.py` - Train a YOLOX model (one call handles everything)
5. `05_evaluate.py` - Run inference on the test set and upload predictions back
6. `06_deploy.py` - Deploy the model as a managed endpoint

## Prerequisites

```bash
pip install datamint
datamint-config --api-key YOUR_API_KEY
```

"""

_SCRIPT_01 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/client_api.html
#
# This is your entry point. Run this script once to upload your images and
# create the Datamint project. After it finishes, open https://app.datamint.io/projects,
# find your project, and draw bounding box annotations on each image before running 04_train.py.

from pathlib import Path
from datamint import Api

PROJECT_NAME = "__PROJECT_NAME__"
DATA_DIR = "./data/images"  # folder containing your images (.jpg, .png, .dcm, ...)

api = Api()

proj = api.projects.create(
    name=PROJECT_NAME,
    description="Object detection project",
    exists_ok=True,  # returns the existing project if it already exists
)

# Upload all images from DATA_DIR and publish them directly into the project.
image_files = [
    f for f in Path(DATA_DIR).iterdir()
    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".dcm")
]
uploaded = api.resources.upload_resources(image_files, publish_to=proj)
print(f"Uploaded {len(uploaded)} images to project '{PROJECT_NAME}'.")

# -------------------------------------------------------------------------
# If you already have bounding box annotations (e.g. from a Pascal VOC or
# COCO export), you can upload them programmatically instead of drawing them
# in the UI. Example for one image:
#
#   api.annotations.add_box_annotation(
#       point1=(x1, y1),
#       point2=(x2, y2),
#       resource=uploaded[0],
#       identifier="class_name",
#   )
#
# Repeat for every box in every image. See the BCCD detection notebook for a
# full example with Pascal VOC XML parsing:
# notebooks/06_end_to_end/slice_based/03_bccd_detection.ipynb
# -------------------------------------------------------------------------

resources = list(proj.fetch_resources())
print(f"Project '{PROJECT_NAME}' now has {len(resources)} images ready for annotation.")
"""

_SCRIPT_02 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/client_api.html
#
# Use this script to inspect your project data and preview annotated examples.
# It prints an annotation summary, then saves explore_samples.png with 2 random
# annotated examples so you can sanity-check your labels before training.

import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from datamint import Api

PROJECT_NAME = "__PROJECT_NAME__"

api = Api()

resources = api.resources.get_list(project_name=PROJECT_NAME)

annotated = []
missing = []
class_counts: dict = defaultdict(int)

for r in resources:
    boxes = r.fetch_annotations(annotation_type="square")
    if boxes:
        annotated.append((r, boxes))
        for box in boxes:
            class_counts[box.identifier] += 1
    else:
        missing.append(r)

print(f"Project        : {PROJECT_NAME}")
print(f"Total images   : {len(resources)}")
print(f"Annotated      : {len(annotated)}")
print(f"Missing labels : {len(missing)}")
if class_counts:
    print("\\nBox counts per class:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
if missing:
    print(f"\\nImages with no annotations ({min(5, len(missing))} shown):")
    for r in missing[:5]:
        print(f"  {r.filename}")

if not annotated:
    print("\\nNo annotated resources found yet. Add bounding boxes in the Datamint UI first.")
else:
    # Sanity check: visualize 2 random annotated examples with boxes overlaid.
    # Open explore_samples.png to verify your labels look correct before training.
    sample = random.sample(annotated, min(2, len(annotated)))
    fig, axes = plt.subplots(1, len(sample), figsize=(7 * len(sample), 6))
    if len(sample) == 1:
        axes = [axes]

    for ax, (resource, boxes) in zip(axes, sample):
        img = resource.fetch_file_data(auto_convert=True)
        img_np = np.array(img) if isinstance(img, Image.Image) else img
        ax.imshow(img_np)
        ax.set_title(resource.filename[:40], fontsize=9)
        ax.axis("off")

        for box in boxes:
            x1, y1, _ = box.geometry.point1
            x2, y2, _ = box.geometry.point2
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none",
            ))
            ax.text(x1, y1 - 4, box.identifier, color="lime", fontsize=8, fontweight="bold")

    plt.suptitle("Annotated examples - sanity check", fontsize=12)
    plt.tight_layout()
    plt.savefig("explore_samples.png", dpi=150)
    plt.close()
    print("\\nSaved explore_samples.png - open it to verify your annotations look correct.")
"""

_SCRIPT_03 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/pytorch_integration.html
#
# build_dataset detects the data type of your project (images, volumes, video)
# and returns the right dataset class automatically.
# box_class_map shows the class names found in your annotations and their integer indices.
# parts.save() persists the split assignments to the Datamint server so 04_train.py
# and 05_evaluate.py can reload the exact same split without recomputing it.

from datamint.dataset import build_dataset

PROJECT_NAME = "__PROJECT_NAME__"

dataset = build_dataset(PROJECT_NAME, return_boxes=True)

print(dataset)
print(f"Detection classes: {dataset.box_class_map}")

# Assign splits in the Datamint UI or programmatically:
#   api.projects.assign_splits(proj, train_resources, "train")
#   api.projects.assign_splits(proj, val_resources,   "val")
#   api.projects.assign_splits(proj, test_resources,  "test")
#
# Or let Datamint split randomly for you (project-scoped, server-side):
#   parts = dataset.split(train=0.7, val=0.15, test=0.15, seed=42)
parts = dataset.split()

train_ds = parts["train"]
val_ds   = parts["val"]
test_ds  = parts["test"]
print(f"Split - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

# Persist the split to the server. 05_evaluate.py will reload it with dataset.split()
# without needing to recompute ratios or seeds.
# If splits already exist, use parts.save(force=True) to overwrite.
parts.save()
print("Split saved to the Datamint project.")
print("Done.")
"""

_SCRIPT_04 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/trainer_api.html
#
# YOLOXTrainer handles the full pipeline in one call:
# dataset loading, augmentation, YOLOX model, MLflow logging, and checkpoint registration.
# The trained model is automatically registered in MLflow under the project name.

from datamint.lightning.trainers.specialized import YOLOXTrainer

PROJECT_NAME = "__PROJECT_NAME__"

# Key parameters to customize:
#   model_size  : 'nano' / 'tiny' / 's' / 'm' / 'l' / 'x' - trade speed vs. accuracy
#   image_size  : input resolution - larger images improve small-object detection
#   batch_size  : reduce if you run out of GPU memory
#   max_epochs  : 50-100 is usually enough for fine-tuning; more for training from scratch
#   conf_thre   : confidence threshold at inference (default 0.25)
#   nms_thre    : IoU threshold for NMS (default 0.45, lower = fewer overlapping boxes)
#   early_stopping_patience : epochs without val improvement before stopping (default 10)
#   mlflow_experiment_name  : name of the MLflow experiment. If not set, it defaults to
#                             "{PROJECT_NAME}_training". Setting it explicitly makes it
#                             easier to find under Training History > Experiments inside
#                             your project on app.datamint.io.
trainer = YOLOXTrainer(
    project=PROJECT_NAME,
    model_size="s",
    image_size=640,
    batch_size=8,
    max_epochs=50,
    accelerator="auto",
    mlflow_experiment_name=PROJECT_NAME,
    # conf_thre=0.25,
    # nms_thre=0.45,
    # early_stopping_patience=10,
)

# -------------------------------------------------------------------------
# Using a different model architecture
#
# If you want to bring your own detection model,
# subclass DetectionTrainer - it handles the dataset (ImageDataset with
# return_boxes=True), detection_collate_fn, and val/map monitoring.
# You only need to implement the model and the albumentations transforms:
#
#   import albumentations as A
#   from albumentations.pytorch import ToTensorV2
#   import lightning as L
#   from datamint.lightning.trainers import DetectionTrainer
#
#   class MyDetector(L.LightningModule):
#       def __init__(self): ...
#       def training_step(self, batch, batch_idx): ...
#       def validation_step(self, batch, batch_idx): ...
#       def test_step(self, batch, batch_idx): ...
#       def configure_optimizers(self): ...
#
#   class MyDetectionTrainer(DetectionTrainer):
#       def _build_model(self, loss_fn, metrics):
#           return MyDetector()
#       def _train_transform(self):
#           return A.Compose(
#               [A.Resize(640, 640), A.HorizontalFlip(p=0.5), A.Normalize(), ToTensorV2()],
#               bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
#           )
#       def _eval_transform(self):
#           return A.Compose(
#               [A.Resize(640, 640), A.Normalize(), ToTensorV2()],
#               bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
#           )
#
#   trainer = MyDetectionTrainer(project=PROJECT_NAME)
# -------------------------------------------------------------------------

print(f"Project              : {PROJECT_NAME}")
print(f"Model                : YOLOX-{trainer.model_size}")
print(f"Image size           : {trainer.image_size}")
print(f"Batch size           : {trainer.batch_size}")
print(f"Max epochs           : {trainer.max_epochs}")
print(f"Early stopping       : {trainer.early_stopping_patience} epochs patience")
print(f"Confidence threshold : {trainer.conf_thre}")
print(f"NMS threshold        : {trainer.nms_thre}")
print()

results = trainer.fit()

print()
print("Training complete.")
metrics = results['test_results'][0] if results['test_results'] else {}
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")
print(f"Model registered as '{PROJECT_NAME}' in MLflow.")
print("Find it under Training History > Experiments inside your project on app.datamint.io.")
"""

_SCRIPT_05 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/trainer_api.html
#
# Reload the test split saved by 03_dataset.py, run the trained model on it,
# upload predictions back to Datamint, and save a local visualization comparing
# ground truth vs. predictions for a few test samples.

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from PIL import Image
import mlflow
from datamint import Api
from datamint.dataset import build_dataset

PROJECT_NAME = "__PROJECT_NAME__"
MODEL_NAME = PROJECT_NAME  # trainer registers the model under the project name by default

api = Api()

# Load the model registered during training
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

# Reload the test split persisted by 03_dataset.py - no ratios or seeds needed
dataset = build_dataset(PROJECT_NAME, return_boxes=True)
parts   = dataset.split()
test_ds = parts["test"]
print(f"Test split: {len(test_ds)} images")

# Run inference and upload predictions back
class_counts: dict = defaultdict(int)
test_resources = list(test_ds.resources)

for resource in test_resources:
    predictions = model.predict([resource])
    box_annotations = predictions[0]
    if box_annotations:
        api.annotations.create(resource, box_annotations)
        for ann in box_annotations:
            class_counts[ann.identifier] += 1

print()
print("Predictions uploaded. Detection summary:")
for cls, count in sorted(class_counts.items()):
    print(f"  {cls}: {count} box(es) across all test images")
print()
print("Open app.datamint.io to compare predictions with ground truth.")

# Visualize ground truth vs. predictions for a few random test samples
sample = random.sample(test_resources, min(3, len(test_resources)))
fig, axes = plt.subplots(len(sample), 2, figsize=(12, 5 * len(sample)))
if len(sample) == 1:
    axes = [axes]

for row, resource in enumerate(sample):
    img = resource.fetch_file_data(auto_convert=True)
    img_np = np.array(img) if isinstance(img, Image.Image) else img

    # Ground truth
    ax_gt = axes[row][0]
    ax_gt.imshow(img_np)
    ax_gt.set_title(f"Ground truth: {resource.filename[:30]}", fontsize=8)
    ax_gt.axis("off")
    for box in resource.fetch_annotations(annotation_type="square"):
        x1, y1, _ = box.geometry.point1
        x2, y2, _ = box.geometry.point2
        ax_gt.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime", facecolor="none",
        ))
        ax_gt.text(x1, y1 - 4, box.identifier, color="lime", fontsize=7, fontweight="bold")

    # Predictions
    ax_pr = axes[row][1]
    ax_pr.imshow(img_np)
    ax_pr.set_title("Prediction", fontsize=8)
    ax_pr.axis("off")
    for ann in model.predict([resource])[0]:
        x1, y1, _ = ann.geometry.point1
        x2, y2, _ = ann.geometry.point2
        ax_pr.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="tomato", facecolor="none",
        ))
        ax_pr.text(x1, y1 - 4, ann.identifier, color="tomato", fontsize=7, fontweight="bold")

plt.suptitle("Ground truth vs. predictions - test samples", fontsize=12)
plt.tight_layout()
plt.savefig("detection_results.png", dpi=150)
plt.close()
print("Saved detection_results.png")
"""

_SCRIPT_06 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/client_api.html
#
# Deploy the registered model as a managed Datamint endpoint.
# Once deployed, inference runs server-side - no local GPU or inference server needed.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from datamint import Api

PROJECT_NAME = "__PROJECT_NAME__"
MODEL_NAME = PROJECT_NAME

api = Api()

# Deploy the model registered during training. This can take a few minutes.
deploy_job = api.deploy.start(model_name=MODEL_NAME, model_alias="latest")
print(f"Deploying '{MODEL_NAME}'... (this may take a few minutes)")

deploy_job = deploy_job.wait()
print(f"Deployment status: {deploy_job.status}")

if deploy_job.status.lower() != "completed":
    msg = deploy_job.error_message or "no details available"
    raise RuntimeError(f"Deployment failed: {msg}")

# Run remote inference - results are saved back to Datamint automatically
resource = api.resources.get_list(project_name=PROJECT_NAME, limit=1)[0]

inf_job = api.inference.submit(
    model_name=MODEL_NAME,
    model_alias="latest",
    resource_id=resource.id,
)
inf_job = inf_job.wait()
print(f"Inference complete for '{resource.filename}'.")

preds = inf_job.predictions[0] if inf_job.predictions else []
print(f"Detections: {len(preds)}")

# Visualize the result
img_np = np.array(resource.fetch_file_data(auto_convert=True, use_cache=True))

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(img_np)
for box in preds:
    x1, y1, _ = box.geometry.point1
    x2, y2, _ = box.geometry.point2
    ax.add_patch(mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="tomato", facecolor="none",
    ))
    ax.text(x1, y1 - 4, box.identifier, color="tomato", fontsize=8, fontweight="bold")
ax.set_title(resource.filename)
ax.axis("off")
plt.tight_layout()
plt.savefig("deploy_result.png", dpi=150)
plt.close()
print("Saved deploy_result.png")
"""


# ---------------------------------------------------------------------------
# Script templates — classification task
# ---------------------------------------------------------------------------

_CLS_README = """\
# __PROJECT_NAME__

Image classification project using Datamint + ImageClassificationTrainer (ResNet-34).

## Getting started

Run the scripts in order:

1. `01_upload_data.py` - Upload images and create the Datamint project
2. `02_explore.py` - Inspect uploaded resources and preview annotated examples
3. `03_dataset.py` - Preview the dataset, class map, and train/val/test splits
4. `04_train.py` - Train a ResNet-34 classifier (one call handles everything)
5. `05_evaluate.py` - Run inference on the test set and upload predictions back
6. `06_deploy.py` - Deploy the model as a managed endpoint

## Prerequisites

```bash
pip install datamint
datamint-config --api-key YOUR_API_KEY
```

"""

_CLS_SCRIPT_01 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/client_api.html
#
# This is your entry point. Organize your images in subfolders named after
# each class (e.g. data/images/cat/, data/images/dog/) and run this script once.
# It uploads all images and creates one image-level classification annotation per
# image so the trainer picks up the labels automatically.

from pathlib import Path
from datamint import Api

PROJECT_NAME = "__PROJECT_NAME__"
DATA_DIR = "./data/images"  # each subfolder is treated as one class label

LABEL_NAME = "label"  # annotation identifier - keep consistent across all scripts

api = Api()

proj = api.projects.create(
    name=PROJECT_NAME,
    description="Image classification project",
    exists_ok=True,  # returns the existing project if it already exists
)

data_root = Path(DATA_DIR)
class_dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]

if not class_dirs:
    raise ValueError(
        f"No class subfolders found in {DATA_DIR}. "
        "Create one subfolder per class, e.g. data/images/cat/, data/images/dog/"
    )

print(f"Found {len(class_dirs)} class(es): {[d.name for d in class_dirs]}")

for class_dir in class_dirs:
    class_name = class_dir.name
    image_files = [
        f for f in class_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".dcm")
    ]
    if not image_files:
        print(f"  Skipping '{class_name}': no images found.")
        continue

    uploaded = api.resources.upload_resources(
        image_files,
        tags=[class_name],
        publish_to=proj,
        progress_bar=True,
    )
    print(f"  '{class_name}': uploaded {len(uploaded)} images.")

    for res in uploaded:
        api.annotations.create_image_classification(
            resource=res,
            identifier=LABEL_NAME,
            value=class_name,
        )

resources = list(proj.fetch_resources())
print(f"Project '{PROJECT_NAME}' now has {len(resources)} images ready for training.")
"""

_CLS_SCRIPT_02 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/client_api.html
#
# Use this script to inspect your project data and preview annotated examples.
# It prints a class distribution summary, then saves explore_samples.png
# with one random image per class so you can sanity-check your labels.

import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datamint import Api

PROJECT_NAME = "__PROJECT_NAME__"

api = Api()

resources = api.resources.get_list(project_name=PROJECT_NAME)

annotated = []
missing = []
class_counts: dict = defaultdict(int)
class_samples: dict = defaultdict(list)  # class_name -> list of resources

for r in resources:
    anns = r.fetch_annotations(annotation_type="category")
    if anns:
        label = anns[0].value
        annotated.append((r, label))
        class_counts[label] += 1
        class_samples[label].append(r)
    else:
        missing.append(r)

print(f"Project        : {PROJECT_NAME}")
print(f"Total images   : {len(resources)}")
print(f"Annotated      : {len(annotated)}")
print(f"Missing labels : {len(missing)}")
if class_counts:
    print("\\nImages per class:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
if missing:
    print(f"\\nImages with no annotations ({min(5, len(missing))} shown):")
    for r in missing[:5]:
        print(f"  {r.filename}")

if not annotated:
    print("\\nNo annotated resources found. Run 01_upload_data.py first.")
else:
    # Sanity check: one random image per class side by side.
    # Open explore_samples.png to verify your labels look correct before training.
    classes = sorted(class_samples.keys())
    sample = [(cls, random.choice(class_samples[cls])) for cls in classes[:6]]
    n = len(sample)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, resource) in zip(axes, sample):
        img = resource.fetch_file_data(auto_convert=True)
        img_np = np.array(img) if isinstance(img, Image.Image) else img
        ax.imshow(img_np)
        ax.set_title(f"{label}\\n{resource.filename[:30]}", fontsize=9)
        ax.axis("off")

    plt.suptitle("One sample per class - sanity check", fontsize=12)
    plt.tight_layout()
    plt.savefig("explore_samples.png", dpi=150)
    plt.close()
    print("\\nSaved explore_samples.png - open it to verify your labels look correct.")
"""

_CLS_SCRIPT_03 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/pytorch_integration.html
#
# build_dataset detects the data type of your project (images, volumes, video)
# and returns the right dataset class automatically.
# image_categories_set lists every (identifier, value) pair found in your annotations.
# parts.save() persists the split to the server so 04_train.py and 05_evaluate.py
# reload the exact same assignment without recomputing ratios or seeds.

from datamint.dataset import build_dataset

PROJECT_NAME = "__PROJECT_NAME__"

dataset = build_dataset(
    PROJECT_NAME,
    include_unannotated=False,
    image_categories_merge_strategy="mode",
    allow_external_annotations=True,
)

print(dataset)
print(f"Classes: {dataset.image_categories_set}")

# Assign splits in the Datamint UI or programmatically:
#   api.projects.assign_splits(proj, train_resources, "train")
#   api.projects.assign_splits(proj, val_resources,   "val")
#   api.projects.assign_splits(proj, test_resources,  "test")
#
# Or let Datamint split randomly for you (project-scoped, server-side):
#   parts = dataset.split(train=0.7, val=0.15, test=0.15, seed=42)
parts = dataset.split()

train_ds = parts["train"]
val_ds   = parts["val"]
test_ds  = parts["test"]
print(f"Split - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

# Persist the split to the server. 04_train.py and 05_evaluate.py will reload it.
# If splits already exist, use parts.save(force=True) to overwrite.
parts.save()
print("Split saved to the Datamint project.")
print("Done.")
"""

_CLS_SCRIPT_04 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/trainer_api.html
#
# ImageClassificationTrainer handles the full pipeline in one call:
# dataset loading, augmentation, ResNet fine-tuning, MLflow logging,
# and checkpoint registration.
# The trained model is automatically registered in MLflow under the project name.

from datamint.lightning.trainers import ImageClassificationTrainer

PROJECT_NAME = "__PROJECT_NAME__"

# Key parameters to customize:
#   model_name  : any timm model (default 'resnet34') - e.g. 'efficientnet_b0', 'vit_small_patch16_224'
#   pretrained  : use ImageNet weights (default True - strongly recommended)
#   image_size  : resize all images to this resolution before training
#   batch_size  : reduce if you run out of GPU memory
#   max_epochs  : 10-30 is usually enough for fine-tuning a pretrained backbone
#   early_stopping_patience : epochs without val improvement before stopping (default 10)
#   mlflow_experiment_name  : name of the MLflow experiment. If not set, it defaults to
#                             "{PROJECT_NAME}_training". Setting it explicitly makes it
#                             easier to find under Training History > Experiments inside
#                             your project on app.datamint.io.
trainer = ImageClassificationTrainer(
    project=PROJECT_NAME,
    model_name="resnet34",
    pretrained=True,
    image_size=224,
    batch_size=16,
    max_epochs=20,
    mlflow_experiment_name=PROJECT_NAME,
    # early_stopping_patience=10,
)

# -------------------------------------------------------------------------
# Using a different model architecture
#
# ImageClassificationTrainer accepts any timm model - just change model_name:
#   trainer = ImageClassificationTrainer(project=PROJECT_NAME, model_name="efficientnet_b3")
#   trainer = ImageClassificationTrainer(project=PROJECT_NAME, model_name="vit_small_patch16_224")
#   trainer = ImageClassificationTrainer(project=PROJECT_NAME, model_name="convnext_tiny")
#
# If you want full control over the model and transforms, subclass
# ClassificationTrainer and implement _build_model, _train_transform, _eval_transform:
#
#   import albumentations as A
#   from albumentations.pytorch import ToTensorV2
#   import lightning as L
#   from datamint.lightning.trainers import ClassificationTrainer
#
#   class MyClassifier(L.LightningModule):
#       def __init__(self): ...
#       def training_step(self, batch, batch_idx): ...
#       def validation_step(self, batch, batch_idx): ...
#       def test_step(self, batch, batch_idx): ...
#       def configure_optimizers(self): ...
#
#   class MyClassificationTrainer(ClassificationTrainer):
#       def _build_model(self, loss_fn, metrics):
#           return MyClassifier()
#       def _train_transform(self):
#           return A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=0.5), A.Normalize(), ToTensorV2()])
#       def _eval_transform(self):
#           return A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
#
#   trainer = MyClassificationTrainer(project=PROJECT_NAME)
# -------------------------------------------------------------------------

print(f"Project              : {PROJECT_NAME}")
print(f"Model                : {trainer.model_name} (pretrained={trainer.pretrained})")
print(f"Image size           : {trainer.image_size}")
print(f"Batch size           : {trainer.batch_size}")
print(f"Max epochs           : {trainer.max_epochs}")
print(f"Early stopping       : {trainer.early_stopping_patience} epochs patience")
print()

results = trainer.fit()

print()
print("Training complete.")
metrics = results['test_results'][0] if results['test_results'] else {}
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")
print(f"Model registered as '{PROJECT_NAME}' in MLflow.")
print("Find it under Training History > Experiments inside your project on app.datamint.io.")
"""

_CLS_SCRIPT_05 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/trainer_api.html
#
# Reload the test split saved by 03_dataset.py, run the trained model on it,
# upload predictions back to Datamint, and save a local visualization comparing
# ground truth vs. predictions for a few test samples.

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mlflow
from datamint import Api
from datamint.dataset import build_dataset

PROJECT_NAME = "__PROJECT_NAME__"
MODEL_NAME = PROJECT_NAME  # trainer registers the model under the project name by default
LABEL_NAME = "label"       # must match the identifier used in 01_upload_data.py

api = Api()

# Load the model registered during training
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

# Reload the test split persisted by 03_dataset.py - no ratios or seeds needed
dataset = build_dataset(
    PROJECT_NAME,
    include_unannotated=False,
    image_categories_merge_strategy="mode",
    allow_external_annotations=True,
)
parts   = dataset.split()
test_ds = parts["test"]
print(f"Test split: {len(test_ds)} images")

# Run inference and upload predictions back
correct = 0
test_resources = list(test_ds.resources)

for resource in test_resources:
    predictions = model.predict([resource])
    pred_anns = predictions[0] if predictions else []

    for ann in pred_anns:
        api.annotations.create_image_classification(
            resource=resource,
            identifier=LABEL_NAME,
            value=ann.value,
        )

    gt_anns = resource.fetch_annotations(annotation_type="category")
    if gt_anns and pred_anns and gt_anns[0].value == pred_anns[0].value:
        correct += 1

accuracy = correct / len(test_resources) if test_resources else 0.0
print(f"Test accuracy: {accuracy:.1%} ({correct}/{len(test_resources)})")
print()
print("Predictions uploaded. Open app.datamint.io to compare with ground truth.")

# Visualize ground truth vs. predictions for a few random test samples
sample = random.sample(test_resources, min(4, len(test_resources)))
n = len(sample)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
if n == 1:
    axes = [axes]

for ax, resource in zip(axes, sample):
    img = resource.fetch_file_data(auto_convert=True)
    img_np = np.array(img) if isinstance(img, Image.Image) else img
    ax.imshow(img_np)
    ax.axis("off")

    gt_anns  = resource.fetch_annotations(annotation_type="category")
    pred_anns = model.predict([resource])[0] if model else []

    gt_label   = gt_anns[0].value if gt_anns else "?"
    pred_label = pred_anns[0].value if pred_anns else "?"
    color = "green" if gt_label == pred_label else "red"
    ax.set_title(f"GT: {gt_label}\\nPred: {pred_label}", fontsize=9, color=color)

plt.suptitle("Ground truth vs. predictions - test samples", fontsize=12)
plt.tight_layout()
plt.savefig("classification_results.png", dpi=150)
plt.close()
print("Saved classification_results.png")
"""

_CLS_SCRIPT_06 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/client_api.html
#
# Deploy the registered model as a managed Datamint endpoint.
# Once deployed, inference runs server-side - no local GPU or inference server needed.

import numpy as np
import matplotlib.pyplot as plt
from datamint import Api

PROJECT_NAME = "__PROJECT_NAME__"
MODEL_NAME = PROJECT_NAME

api = Api()

# Deploy the model registered during training. This can take a few minutes.
deploy_job = api.deploy.start(model_name=MODEL_NAME, model_alias="latest")
print(f"Deploying '{MODEL_NAME}'... (this may take a few minutes)")

deploy_job = deploy_job.wait()
print(f"Deployment status: {deploy_job.status}")

if deploy_job.status.lower() != "completed":
    msg = deploy_job.error_message or "no details available"
    raise RuntimeError(f"Deployment failed: {msg}")

# Run remote inference - results are saved back to Datamint automatically
resource = api.resources.get_list(project_name=PROJECT_NAME, limit=1)[0]

inf_job = api.inference.submit(
    model_name=MODEL_NAME,
    model_alias="latest",
    resource_id=resource.id,
)
inf_job = inf_job.wait()
print(f"Inference complete for '{resource.filename}'.")

preds = inf_job.predictions[0] if inf_job.predictions else []
label = preds[0].value if preds else "no prediction"
print(f"Prediction: {label}")

# Visualize the result
img_np = np.array(resource.fetch_file_data(auto_convert=True, use_cache=True))

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_np)
ax.set_title(f"{resource.filename}\\nPrediction: {label}", fontsize=10)
ax.axis("off")
plt.tight_layout()
plt.savefig("deploy_result.png", dpi=150)
plt.close()
print("Saved deploy_result.png")
"""


# ---------------------------------------------------------------------------
# Script templates — segmentation task
# ---------------------------------------------------------------------------

_SEG_README = """\
# __PROJECT_NAME__

Semantic segmentation project using Datamint + UNet++.

## Getting started

Run the scripts in order:

1. `01_upload_data.py` - Upload images and masks, create the Datamint project
2. `02_explore.py` - Inspect uploaded resources and preview annotated examples
3. `03_dataset.py` - Preview the dataset, segmentation classes, and train/val/test splits
4. `04_train.py` - Train a UNet++ model (one call handles everything)
5. `05_evaluate.py` - Run inference on the test set and upload predictions back
6. `06_deploy.py` - Deploy the model as a managed endpoint

## Prerequisites

```bash
pip install datamint
datamint-config --api-key YOUR_API_KEY
```

"""

_SEG_SCRIPT_01 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/client_api.html
#
# This is your entry point. Place your images in DATA_DIR and the matching
# segmentation masks in MASKS_DIR. Files are paired by name - image
# "case001.png" must match mask "case001.png" (or "case001_mask.png").
# Masks must be grayscale PNGs where non-zero pixels mark the foreground.
#
# For 3D volumes (NIfTI, DICOM series):
#   Upload volume files the same way - upload_resources handles them automatically.
#   Replace upload_segmentations with upload_volume_segmentation for the masks.

from pathlib import Path
from datamint import Api

PROJECT_NAME = "__PROJECT_NAME__"
DATA_DIR   = "./data/images"   # directory containing the input images
MASKS_DIR  = "./data/masks"    # directory containing the binary mask PNGs
LABEL_NAME = "foreground"      # class name given to every mask

api = Api()

proj = api.projects.create(
    name=PROJECT_NAME,
    description="Semantic segmentation project",
    exists_ok=True,
)

image_files = sorted([
    f for f in Path(DATA_DIR).iterdir()
    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".dcm", ".nii", ".gz")
])
if not image_files:
    raise FileNotFoundError(f"No images found in {DATA_DIR}.")

uploaded = api.resources.upload_resources(image_files, publish_to=proj, progress_bar=True)
print(f"Uploaded {len(uploaded)} images.")

masks_dir = Path(MASKS_DIR)
missing_masks = 0
for res in uploaded:
    mask_path = masks_dir / res.filename
    if not mask_path.exists():
        stem = Path(res.filename).stem
        for suffix in (".png", "_mask.png", "_seg.png"):
            candidate = masks_dir / f"{stem}{suffix}"
            if candidate.exists():
                mask_path = candidate
                break

    if mask_path.exists():
        api.annotations.upload_segmentations(
            resource=res,
            file_path=mask_path,
            name=LABEL_NAME,
        )
    else:
        missing_masks += 1

if missing_masks:
    print(f"Warning: {missing_masks} image(s) had no matching mask - skipped.")

resources = list(proj.fetch_resources())
print(f"Project '{PROJECT_NAME}' now has {len(resources)} images ready for training.")
"""

_SEG_SCRIPT_02 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/client_api.html
#
# Use this script to inspect your project data and preview annotated examples.
# It prints annotation coverage and saves explore_samples.png with a few
# images overlaid with their ground-truth segmentation masks.

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datamint import Api

PROJECT_NAME = "__PROJECT_NAME__"

api = Api()
resources = list(api.resources.get_list(project_name=PROJECT_NAME))

annotated = []
missing = []
label_counts: dict = {}

for r in resources:
    anns = r.fetch_annotations(annotation_type="segmentation")
    if anns:
        annotated.append((r, anns))
        for ann in anns:
            label_counts[ann.name] = label_counts.get(ann.name, 0) + 1
    else:
        missing.append(r)

print(f"Project          : {PROJECT_NAME}")
print(f"Total images     : {len(resources)}")
print(f"With masks       : {len(annotated)}")
print(f"Without masks    : {len(missing)}")
if label_counts:
    print("\\nInstances per class:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
if missing:
    print(f"\\nImages with no masks ({min(5, len(missing))} shown):")
    for r in missing[:5]:
        print(f"  {r.filename}")

if not annotated:
    print("\\nNo annotated resources found. Run 01_upload_data.py first.")
else:
    # One random annotated image per slot, overlaid with its mask(s).
    # Open explore_samples.png to confirm masks align with the images.
    sample = random.sample(annotated, min(4, len(annotated)))
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]

    fig, axes = plt.subplots(1, len(sample), figsize=(6 * len(sample), 5))
    if len(sample) == 1:
        axes = [axes]

    def _to_2d_mask(data):
        if isinstance(data, np.ndarray):
            arr = data
        elif hasattr(data, 'get_fdata'):  # NIfTI
            arr = np.squeeze(data.get_fdata())
        else:
            arr = np.array(data)
        if arr.ndim == 3:
            arr = arr[..., arr.shape[-1] // 2]  # middle slice
        return arr

    for ax, (resource, anns) in zip(axes, sample):
        img = resource.fetch_file_data(auto_convert=True)
        img_np = np.array(img) if isinstance(img, Image.Image) else img
        ax.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)

        for i, ann in enumerate(anns):
            mask_data = ann.fetch_file_data(auto_convert=True)
            mask = _to_2d_mask(mask_data)
            if mask.shape != img_np.shape[:2]:
                mask = mask.T  # NIfTI axes are (W, H); numpy expects (H, W)
            overlay = np.zeros((*img_np.shape[:2], 4))
            overlay[..., :3] = colors[i % len(colors)]
            overlay[..., 3] = 0.4 * (mask > 0)
            ax.imshow(overlay)

        label_names = ", ".join(a.name for a in anns)
        ax.set_title(f"{resource.filename[:25]}\\n{label_names}", fontsize=8)
        ax.axis("off")

    plt.suptitle("Segmentation masks overlay - sanity check", fontsize=12)
    plt.tight_layout()
    plt.savefig("explore_samples.png", dpi=150)
    plt.close()
    print("\\nSaved explore_samples.png - open it to confirm masks align correctly.")
"""

_SEG_SCRIPT_03 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/pytorch_integration.html
#
# build_dataset detects the data type of your project and returns the right
# dataset class automatically. segmentation_labels_set lists every foreground
# class in your annotations. parts.save() persists the split to the server so
# 04_train.py and 05_evaluate.py reload the exact same assignment.
#
# For 3D volumes, SemanticSegmentation2DTrainer slices volumes automatically -
# no code change is needed here. If you want patient-wise splits instead:
#   parts = dataset.split(train=0.7, val=0.15, test=0.15, seed=42, by_patient=True)

from datamint.dataset import build_dataset

PROJECT_NAME = "__PROJECT_NAME__"

dataset = build_dataset(
    PROJECT_NAME,
    return_as_semantic_segmentation=True,
    semantic_seg_merge_strategy="union",
    allow_external_annotations=True,
    include_unannotated=False,
)

print(dataset)
print(f"Segmentation classes: {dataset.segmentation_labels_set}")

# Assign splits in the Datamint UI or programmatically:
#   api.projects.assign_splits(proj, train_resources, "train")
#   api.projects.assign_splits(proj, val_resources,   "val")
#   api.projects.assign_splits(proj, test_resources,  "test")
#
# Or let Datamint split randomly for you (project-scoped, server-side):
#   parts = dataset.split(train=0.7, val=0.15, test=0.15, seed=42)
parts = dataset.split()

train_ds = parts["train"]
val_ds   = parts["val"]
test_ds  = parts["test"]
print(f"Split - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

# Persist the split to the server. 04_train.py and 05_evaluate.py will reload it.
# If splits already exist, use parts.save(force=True) to overwrite.
parts.save()
print("Split saved to the Datamint project.")
print("Done.")
"""

_SEG_SCRIPT_04 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/trainer_api.html
#
# UNetPPTrainer handles the full 2D segmentation pipeline in one call:
# dataset loading, augmentation, UNet++ training with BCE + Dice loss,
# IoU + Dice metrics, MLflow logging, early stopping, and model registration.
# The trained model is automatically registered in MLflow under the project name.
#
# For 3D volumes:
#   SemanticSegmentation2DTrainer auto-slices volumes - no code change needed
#   for slice-based training. For true volumetric (patch-based) training, use:
#     from datamint.lightning.trainers import UNETRPPTrainer
#     trainer = UNETRPPTrainer(project=PROJECT_NAME, ...)
#   or nnUNet (automatic architecture search, strongest 3D baseline):
#     from datamint.lightning.trainers import NNUNetTrainer
#     trainer = NNUNetTrainer(project=PROJECT_NAME, ...)

from datamint.lightning.trainers import UNetPPTrainer

PROJECT_NAME = "__PROJECT_NAME__"

# Key parameters to customize:
#   encoder_name : segmentation_models_pytorch encoder (default 'resnet34')
#   image_size   : resize all images to this resolution before training
#   batch_size   : reduce if you run out of GPU memory
#   max_epochs   : 50 is usually enough with early stopping
#   early_stopping_patience : epochs without val improvement before stopping (default 10)
#   mlflow_experiment_name  : name of the MLflow experiment. If not set, it defaults to
#                             "{PROJECT_NAME}_training". Setting it explicitly makes it
#                             easier to find under Training History > Experiments inside
#                             your project on app.datamint.io.
trainer = UNetPPTrainer(
    project=PROJECT_NAME,
    image_size=256,
    batch_size=16,
    max_epochs=50,
    mlflow_experiment_name=PROJECT_NAME,
    # encoder_name="efficientnet-b4",  # larger encoder for harder tasks
    # early_stopping_patience=10,
)

# -------------------------------------------------------------------------
# Alternative 2D model architectures
#
# DeepLab V3+ (atrous convolutions, stronger on large objects):
#   from datamint.lightning.trainers import DeepLabV3PlusTrainer
#   trainer = DeepLabV3PlusTrainer(project=PROJECT_NAME, image_size=256, batch_size=16)
#
# TransUNet (Vision Transformer encoder, better global context):
#   from datamint.lightning.trainers import TransUNetTrainer
#   trainer = TransUNetTrainer(project=PROJECT_NAME, batch_size=8)
#   # TransUNetTrainer always uses 224x224; do not pass image_size.
#
# Custom architecture (keep Datamint loss, metrics, and deployment adapter):
#   from datamint.lightning import SemanticSegmentation2DTrainer
#   from datamint.lightning.trainers.lightning_modules import SegmentationModule
#   import segmentation_models_pytorch as smp
#
#   class MyModel(SegmentationModule):
#       def __init__(self, *args, **kwargs):
#           super().__init__(*args, class_names=["foreground"], **kwargs)
#           self.model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1)
#       def forward(self, x):
#           return self.model(x)
#
#   trainer = SemanticSegmentation2DTrainer(project=PROJECT_NAME, model=MyModel, image_size=256)
# -------------------------------------------------------------------------

print(f"Project       : {PROJECT_NAME}")
print(f"Image size    : {trainer.image_size}")
print(f"Batch size    : {trainer.batch_size}")
print(f"Max epochs    : {trainer.max_epochs}")
print()

results = trainer.fit()

print()
print("Training complete.")
metrics = results["test_results"][0] if results.get("test_results") else {}
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")
print(f"Model registered as '{PROJECT_NAME}' in MLflow.")
print("Find it under Training History > Experiments inside your project on app.datamint.io.")
"""

_SEG_SCRIPT_05 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/trainer_api.html
#
# Reload the test split saved by 03_dataset.py, run the trained model on it,
# upload predictions back to Datamint, and save a local visualization comparing
# ground truth vs. predictions for a few test samples.
#
# model.predict() returns ImageSegmentation objects with a .mask (numpy array).
# api.annotations.upload_segmentations accepts a numpy array directly.

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datamint import Api
from datamint.dataset import build_dataset
from datamint.mlflow import flavors as datamint_flavor

PROJECT_NAME = "__PROJECT_NAME__"
MODEL_NAME   = PROJECT_NAME  # trainer registers the model under the project name by default

api = Api()

# Load the model registered during training
model = datamint_flavor.load_model(f"models:/{MODEL_NAME}/latest")

# Reload the test split persisted by 03_dataset.py - no ratios or seeds needed
dataset = build_dataset(
    PROJECT_NAME,
    return_as_semantic_segmentation=True,
    semantic_seg_merge_strategy="union",
    allow_external_annotations=True,
    include_unannotated=False,
)
parts          = dataset.split()
test_ds        = parts["test"]
test_resources = list(test_ds.resources)
print(f"Test split: {len(test_resources)} images")

# Run inference and upload predictions back
for resource in test_resources:
    pred_anns = model.predict([resource])[0]
    for ann in pred_anns:
        api.annotations.upload_segmentations(
            resource=resource,
            file_path=ann.mask,
            name=ann.name,
        )

print("Predictions uploaded. Open app.datamint.io to compare with ground truth.")
print()

# Visualize ground truth vs. predictions side by side for a few test samples
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]


def _get_mask_np(ann):
    data = ann.fetch_file_data(auto_convert=True)
    if isinstance(data, np.ndarray):
        arr = data
    elif hasattr(data, 'get_fdata'):  # NIfTI
        arr = np.squeeze(data.get_fdata())
    else:
        arr = np.array(data)
    if arr.ndim == 3:
        arr = arr[..., arr.shape[-1] // 2]  # middle slice
    return arr > 0


def _overlay(ax, img_np, anns):
    ax.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
    for i, ann in enumerate(anns):
        mask = _get_mask_np(ann)
        if mask.shape != img_np.shape[:2]:
            mask = mask.T  # NIfTI axes are (W, H); numpy expects (H, W)
        rgba = np.zeros((*img_np.shape[:2], 4))
        rgba[..., :3] = colors[i % len(colors)]
        rgba[..., 3] = 0.4 * mask
        ax.imshow(rgba)
    ax.axis("off")


sample = random.sample(test_resources, min(4, len(test_resources)))
fig, axes = plt.subplots(len(sample), 2, figsize=(10, 5 * len(sample)))
if len(sample) == 1:
    axes = [axes]

for row_axes, resource in zip(axes, sample):
    img = resource.fetch_file_data(auto_convert=True)
    img_np = np.array(img) if isinstance(img, Image.Image) else img

    gt_anns   = resource.fetch_annotations(annotation_type="segmentation")
    pred_anns = model.predict([resource])[0]

    _overlay(row_axes[0], img_np, gt_anns)
    row_axes[0].set_title(f"GT: {resource.filename[:30]}", fontsize=8)

    _overlay(row_axes[1], img_np, pred_anns)
    row_axes[1].set_title("Prediction", fontsize=8)

plt.suptitle("Ground truth vs. predictions - test samples", fontsize=12)
plt.tight_layout()
plt.savefig("segmentation_results.png", dpi=150)
plt.close()
print("Saved segmentation_results.png")
"""

_SEG_SCRIPT_06 = """\
# Docs: https://sonanceai.github.io/datamint-python-api/client_api.html
#
# Deploy the registered model as a managed Datamint endpoint.
# Once deployed, inference runs server-side - no local GPU or inference server needed.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datamint import Api

PROJECT_NAME = "__PROJECT_NAME__"
MODEL_NAME   = PROJECT_NAME

api = Api()

# Deploy the model registered during training. This can take a few minutes.
deploy_job = api.deploy.start(model_name=MODEL_NAME, model_alias="latest")
print(f"Deploying '{MODEL_NAME}'... (this may take a few minutes)")

deploy_job = deploy_job.wait()
print(f"Deployment status: {deploy_job.status}")

if deploy_job.status.lower() != "completed":
    msg = deploy_job.error_message or "no details available"
    raise RuntimeError(f"Deployment failed: {msg}")

# Run remote inference on one image - predictions are saved back to Datamint automatically
resource = api.resources.get_list(project_name=PROJECT_NAME, limit=1)[0]

inf_job = api.inference.submit(
    model_name=MODEL_NAME,
    model_alias="latest",
    resource_id=resource.id,
)
inf_job = inf_job.wait()
print(f"Inference complete for '{resource.filename}'.")

# Visualize the image alongside each predicted mask
preds = inf_job.predictions[0] if inf_job.predictions else []
img_np = np.array(resource.fetch_file_data(auto_convert=True, use_cache=True))

n_plots = 1 + len(preds)
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
if n_plots == 1:
    axes = [axes]

axes[0].imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
axes[0].set_title(resource.filename[:30], fontsize=9)
axes[0].axis("off")

for i, pred in enumerate(preds):
    mask = np.array(pred.fetch_file_data(auto_convert=True))
    if mask.ndim == 3:
        mask = mask[..., 0]
    axes[i + 1].imshow(mask, cmap="gray")
    axes[i + 1].set_title(f"Mask: {pred.name}", fontsize=9)
    axes[i + 1].axis("off")

plt.tight_layout()
plt.savefig("deploy_result.png", dpi=150)
plt.close()
print("Saved deploy_result.png")
"""


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _render(template: str, project_name: str) -> str:
    return template.replace("__PROJECT_NAME__", project_name)


def _generate_detection_files(project_name: str) -> dict[str, str]:
    return {
        "README.md":         _render(_README, project_name),
        "01_upload_data.py": _render(_SCRIPT_01, project_name),
        "02_explore.py":     _render(_SCRIPT_02, project_name),
        "03_dataset.py":     _render(_SCRIPT_03, project_name),
        "04_train.py":       _render(_SCRIPT_04, project_name),
        "05_evaluate.py":    _render(_SCRIPT_05, project_name),
        "06_deploy.py":      _render(_SCRIPT_06, project_name),
    }


def _generate_classification_files(project_name: str) -> dict[str, str]:
    return {
        "README.md":         _render(_CLS_README, project_name),
        "01_upload_data.py": _render(_CLS_SCRIPT_01, project_name),
        "02_explore.py":     _render(_CLS_SCRIPT_02, project_name),
        "03_dataset.py":     _render(_CLS_SCRIPT_03, project_name),
        "04_train.py":       _render(_CLS_SCRIPT_04, project_name),
        "05_evaluate.py":    _render(_CLS_SCRIPT_05, project_name),
        "06_deploy.py":      _render(_CLS_SCRIPT_06, project_name),
    }


def _generate_segmentation_files(project_name: str) -> dict[str, str]:
    return {
        "README.md":         _render(_SEG_README, project_name),
        "01_upload_data.py": _render(_SEG_SCRIPT_01, project_name),
        "02_explore.py":     _render(_SEG_SCRIPT_02, project_name),
        "03_dataset.py":     _render(_SEG_SCRIPT_03, project_name),
        "04_train.py":       _render(_SEG_SCRIPT_04, project_name),
        "05_evaluate.py":    _render(_SEG_SCRIPT_05, project_name),
        "06_deploy.py":      _render(_SEG_SCRIPT_06, project_name),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_header() -> None:
    console.print()
    console.rule("[bold]datamint-init[/bold]")
    console.print()
    console.print(
        " This command generates a set of Python scripts that walk you\n"
        " through a complete Datamint workflow:\n"
    )
    console.print("   [bold]01[/bold]  Upload your data")
    console.print("   [bold]02[/bold]  Explore resources and annotations")
    console.print("   [bold]03[/bold]  Build a PyTorch dataset")
    console.print("   [bold]04[/bold]  Train a model")
    console.print("   [bold]05[/bold]  Evaluate and push predictions back")
    console.print("   [bold]06[/bold]  Deploy as a managed endpoint")
    console.print()
    console.print(
        " Answer a few questions and the scripts will be tailored to your task."
    )
    console.print()
    console.print(Rule())
    console.print()


def main() -> None:
    _print_header()

    try:
        project_name = Prompt.ask(" Project name", console=console).strip()
        if not project_name:
            console.print("[red] Project name cannot be empty.[/red]")
            sys.exit(1)

        task = Prompt.ask(
            " Task",
            choices=["detection", "segmentation", "classification"],
            console=console,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print()
        sys.exit(0)

    out_dir = Path(project_name)
    if out_dir.exists() and any(out_dir.iterdir()):
        console.print(
            f"\n [yellow]Directory './{project_name}/' already exists and is not empty.[/yellow]"
        )
        try:
            if not Confirm.ask(" Overwrite?", default=False, console=console):
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            console.print()
            sys.exit(0)

    out_dir.mkdir(exist_ok=True)
    if task == "classification":
        files = _generate_classification_files(project_name)
    elif task == "segmentation":
        files = _generate_segmentation_files(project_name)
    else:
        files = _generate_detection_files(project_name)

    console.print()
    console.print(f" Generating scripts in [bold]./{project_name}/[/bold] ...")
    console.print()
    for filename, content in files.items():
        (out_dir / filename).write_text(content, encoding="utf-8")
        console.print(f"   [green]✓[/green] {filename}")

    console.print()
    console.print(
        f" [bold green]All set![/bold green]"
        f" Open [bold]{project_name}/[/bold] and start with"
        f" [bold]01_upload_data.py[/bold]"
    )
    console.print()


if __name__ == "__main__":
    main()
