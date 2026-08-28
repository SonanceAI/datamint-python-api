.. _inference:


Bringing an External Model into Datamint
==========================================

:bdg-warning:`Intermediate`

This page is for models **trained entirely outside Datamint**, your own training loop,
Keras, Hugging Face, or any third-party framework. It walks through bringing such a
model in, logging it to MLflow, and deploying it for inference through the UI.

Two ways to integrate, depending on what you have:

- Your model fits a supported task family and you have a labeled Datamint project to
  evaluate it against, and you want metrics logged for the team? Use the
  :ref:`Shortcut <external_model_shortcut>` below.
- Otherwise, any model, any task, no labeled data required? Use the
  :ref:`Custom Adapter <external_model_custom_adapter>` steps below.

Both paths end at the same :ref:`Deploy <external_model_deploy>` step.

If you want to **train** an external model using Datamint, see
:ref:`Training an External Model Through a Datamint Trainer <training_external_model>`,
which covers swapping in your own architecture.


.. _external_model_shortcut:

Shortcut: Your Model Fits a Supported Task Family
----------------------------------------------------

If your model can be expressed as a ``SegmentationModule``/``ClassificationModule``
subclass (the same task-family shapes used by :doc:`trainer_api`) and you already have
a Datamint project with an annotated test split, skip the custom adapter below: wrap
your pretrained weights as a Lightning module and call
``trainer.test(register_model=True)``. It runs zero training epochs (your weights are
untouched), computes test metrics, and registers the model in MLflow in one call (useful when you want metrics 
logged for the team alongside the model.)

.. code-block:: python

   import segmentation_models_pytorch as smp
   import torch

   from datamint.lightning import SemanticSegmentation2DTrainer
   from datamint.lightning.trainers.lightning_modules import SegmentationModule

   net = smp.UnetPlusPlus(encoder_name='resnet34', in_channels=3, classes=1)
   net.load_state_dict(torch.load('my_checkpoint.pth', map_location='cpu'))


   class ExternalSegModule(SegmentationModule):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, class_names=['lesion'], **kwargs)
           self.model = net

       def forward(self, x):
           return self.model(x)


   MODEL_NAME = "my-external-unet"

   trainer = SemanticSegmentation2DTrainer(
       project="MyProject",
       image_size=256,
       model=ExternalSegModule,
       model_name=MODEL_NAME,
   )
   test_metrics = trainer.test(register_model=True)

``test_metrics`` is logged to the MLflow run and shows up in the Datamint dashboard
alongside metrics from any trainer-trained model. Skip ahead to
:ref:`Deploy <external_model_deploy>` -- no manual ``log_model()``/adapter needed for
this path. Note that ``register_model=True`` does not set a ``champion`` alias, so use
``model_version=`` when deploying.

This shortcut doesn't apply if your model doesn't fit one of Datamint's task families,
or you have no labeled test split to evaluate against, use the custom adapter path
below instead.


.. _external_model_custom_adapter:

Custom Adapter: Any Model, No Labeled Data Required
--------------------------------------------------------

The steps below build a generic adapter that works for any model, task, or dataset,
no metrics, no labeled project required. Use them when the shortcut above doesn't
apply.

Load Your Checkpoint
~~~~~~~~~~~~~~~~~~~~~~

Load your model exactly as you would outside Datamint.

.. code-block:: python

   import segmentation_models_pytorch as smp
   import torch

   net = smp.UnetPlusPlus(encoder_name='resnet34', in_channels=3, classes=1)
   state = torch.load('my_checkpoint.pth', map_location='cpu')
   net.load_state_dict(state)
   net.eval()

If the checkpoint is a Lightning ``.ckpt`` produced by one of Datamint's own modules
(e.g. ``SegmentationModule``), use Lightning's own loader instead:
``SegmentationModule.load_from_checkpoint(path)``.

Wrap It in a ``DatamintModel`` Adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Datamint's prediction contract expects a list of
:py:class:`~datamint.entities.resource.BaseResource` objects in, and a
``list[list[Annotation]]`` out -- one annotation list per resource. Subclass
:py:class:`~datamint.mlflow.flavors.model.DatamintModel` and implement
``predict_default`` to bridge your model to that contract:

.. code-block:: python

   import albumentations as A
   import cv2
   import numpy as np
   import torch
   from albumentations.pytorch import ToTensorV2

   from datamint.entities.annotations import ImageSegmentation
   from datamint.mlflow.flavors.model import DatamintModel, ModelSettings
   from datamint.mlflow.flavors.task_type import TaskType


   class SegmentationAdapter(DatamintModel):
       """Wraps a plain nn.Module for Datamint segmentation inference."""

       task_type = TaskType.IMAGE_SEGMENTATION

       def __init__(self, torch_model, class_names, image_size=256, threshold=0.5, need_gpu=False):
           super().__init__(torch_model=torch_model, settings=ModelSettings(need_gpu=need_gpu))
           self.class_names = class_names
           self.image_size = image_size
           self.threshold = threshold
           self._transform = A.Compose([
               A.Resize(image_size, image_size),
               A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
               ToTensorV2(),
           ])

       def predict_default(self, model_input, **kwargs):
           device = self.inference_device
           model = self.get_pytorch_model().to(device).eval()

           results = []
           for resource in model_input:
               img = np.array(resource.fetch_file_data(auto_convert=True, use_cache=True))
               if img.ndim == 2:
                   img = np.stack([img, img, img], axis=-1)
               elif img.shape[-1] == 4:
                   img = img[..., :3]
               orig_h, orig_w = img.shape[:2]

               tensor = self._transform(image=img)['image'].unsqueeze(0).to(device)
               with torch.inference_mode():
                   logits = model(tensor)
               probs = logits.sigmoid().squeeze(0).cpu().numpy()

               results.append([
                   ImageSegmentation(
                       name=self.class_names[i],
                       segmentation_data=cv2.resize(
                           (probs[i] > self.threshold).astype(np.uint8),
                           (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST,
                       ),
                   )
                   for i in range(len(self.class_names))
               ])
           return results


   adapter = SegmentationAdapter(torch_model=net, class_names=['lesion'], image_size=256)

Smoke-Test the Adapter Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before logging anything to MLflow, call ``.predict()`` directly to make sure
``predict_default`` runs without errors:

.. code-block:: python

   import io
   from PIL import Image
   from datamint.entities.resource import LocalResource

   buf = io.BytesIO()
   Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)).save(buf, format='PNG')
   dummy_resource = LocalResource(raw_data=buf.getvalue(), filename='dummy.png')

   predictions = adapter.predict([dummy_resource])
   for ann in predictions[0]:
       print(f"{ann.name!r}  mask shape={ann.mask.shape}")


Log & Register the Model in MLflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``datamint.mlflow.flavors.log_model`` serialises your adapter (including the embedded
``nn.Module``) as an MLflow artifact, records ``task_type`` so the Datamint server knows
how to display predictions, and registers it in the Model Registry under a name you
choose. Calling ``datamint.mlflow.set_project`` first points MLflow at your Datamint
server and associates the run with the right project, so it shows up in the Datamint
dashboard.

.. code-block:: python

   import mlflow

   import datamint.mlflow as datamint_mlflow
   from datamint.mlflow.flavors import log_model

   PROJECT_NAME = "MyProject"
   MODEL_NAME = "my-external-unet"

   datamint_mlflow.set_project(PROJECT_NAME)
   mlflow.set_experiment(PROJECT_NAME)

   with mlflow.start_run(run_name='external_model_upload') as run:
       mlflow.log_params({
           'encoder': 'resnet34',
           'image_size': 256,
           'framework': 'segmentation_models_pytorch',
       })
       model_info = log_model(
           adapter,
           task_type=TaskType.IMAGE_SEGMENTATION,
           name='segmentation_model',
           registered_model_name=MODEL_NAME,
       )

   print(f"Model URI : {model_info.model_uri}")

Assign an Alias
^^^^^^^^^^^^^^^^^

Deployment resolves models by alias, not raw version number. Set one (commonly
``champion``) on the version you just registered:

.. code-block:: python

   from mlflow import MlflowClient

   client = MlflowClient()
   versions = client.search_model_versions(f"name='{MODEL_NAME}'")
   latest_version = max(versions, key=lambda v: int(v.version))
   client.set_registered_model_alias(MODEL_NAME, 'champion', latest_version.version)

At this point you can already verify the round trip by loading the model back and
predicting with it, exactly as you would for a trainer-registered model:

.. code-block:: python

   from datamint import Api

   api = Api()
   model = api.models.get_by_name(MODEL_NAME)
   loaded_model = model.get_latest_version(alias='champion').load_model()

   resources = list(api.resources.get_list(project_name=PROJECT_NAME, limit=1))
   predictions = loaded_model.predict(resources)


.. _external_model_deploy:

Deploy: Run Inference Through the UI
------------------------------------------

Deploying starts a serving instance for the registered model so predictions can be
triggered directly from the Datamint platform, without writing any code per prediction.
Resolve by ``model_alias`` if you set one (the custom adapter path above does this), or
by ``model_version`` directly if you came from the shortcut, which doesn't set an alias:

.. code-block:: python

   job = api.deploy.start(
       model_name=MODEL_NAME,
       model_alias='champion',   # or: model_version=1
       with_gpu=False,
   )

   import time
   while True:
       job = api.deploy.get_by_id(job.id)
       if job.status in ('completed', 'failed', 'cancelled'):
           break
       time.sleep(15)

Once ``job.status == 'completed'``, the model is available for inference from the
Datamint UI on any resource in the project. The same call is also available
programmatically, e.g. for batch/automated inference, via ``api.inference.submit``:

.. code-block:: python

   inf_job = api.inference.submit(
       model_name=MODEL_NAME,
       model_alias='champion',
       resource_id=resources[0].id,
   )
   inf_job.wait()
   predictions = inf_job.predictions


Related Examples
------------------

- `Deploying an Externally Trained Model <https://github.com/SonanceAI/datamint-python-api/blob/main/notebooks/05_deployment/02_deploy_external_model.ipynb>`_
  -- the full runnable notebook this page is based on, including custom prediction
  modes and updating a deployed model with a new version.
