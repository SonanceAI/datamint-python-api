---
name: datamint-deploy
description: Deploy registered Datamint models and run inference with the `datamint` SDK - starting and monitoring deployment jobs, choosing the model version, GPU deployments, managing deployed images, submitting remote inference jobs on resources (images, frames, slices, volumes, promptable models), saving predictions back as annotations, reading serving-pod logs, running a registered model locally (`load_model`, `datamint inference` CLI) and testing a model in a local podman container before deploying. Use this whenever the user wants to deploy a model, serve it, run predictions with a Datamint model, automate inference on a project, or debug a failed deployment or inference job.
---

# Deploying models and running inference

```python
from datamint import Api
api = Api()
```

A model must be **registered** before it can be deployed (built-in trainers and `api.models.log_model` register it; see `datamint-training`, `datamint-custom-models`). Deploying builds a serving image for one registered version, so predictions can run on the server and from the Datamint UI.

## Where to run predictions

| Need | Use |
|---|---|
| Quick check or local batch, no deployment | `api.models.get_by_name(name).get_latest_version().load_model().predict(resources)` |
| One local file from the terminal | `datamint inference file.png --model-name MyModel` |
| Test the serving image on own machine before deploying | Local podman build (end of this skill) |
| Server-side inference, UI predictions, automation | Deploy, then `api.inference.submit(...)` |

## Deploying

```python
job = api.deploy.start(model_name='unetpp_r34', with_gpu=False)
job.wait(timeout=1800)                   # returns when finished; raises TimeoutError only on timeout
if job.status != 'completed':
    print(job.error_message)
    print(job.get_logs(tail=200))        # Docker build logs
```

Version selection (pass at most one):
- Neither `model_version` nor `model_alias`: deploys the **highest registered version** and tags the image `latest`. This is the default, and inference without a version or alias uses this image.
- `model_version=3`: that exact version.
- `model_alias='prod'`: the version an alias points to. The alias must exist on the registered model (`latest` is reserved by MLflow and can't be set as an alias).

Other options: `with_gpu=True` for models that need a GPU, `image_name=` to name the image.

- `wait()` doesn't raise when the job fails. Check `job.status` (`'completed'`, `'failed'`, `'cancelled'`, `'error'`) afterwards.
- `job.wait(on_log=print)` streams build log lines. `on_status=` receives status updates.
- Deploying a new version: register it, then call `start` again. A default deploy picks up the new highest version.
- Builds take minutes. `api.deploy.get_by_id(job.id)` gives `status`, `progress_percentage`, `current_step`, `recent_logs`, `image_name`, `image_tag`.

Managing deployments:

```python
api.deploy.cancel(job)
api.deploy.list_active_jobs()
api.deploy.list_images(model_name='unetpp_r34')          # [{'name', 'tag', ...}, ...]
api.deploy.image_exists('unetpp_r34')                    # tag defaults to 'latest'
api.models.get_by_name('unetpp_r34').is_deployed()
api.deploy.remove_image('unetpp_r34', tag='latest')      # without tag: removes every image of the model
```

`remove_image` is irreversible for the serving image. Confirm with the user first.

## Remote inference

```python
inf = api.inference.submit(model_name='unetpp_r34', resource_ids=[r.id for r in resources])
inf.wait()
if inf.status == 'completed':
    preds = inf.predictions          # list[list[Annotation]], one list per input resource
else:
    print(inf.error_message)
    print(inf.get_pod_logs(tail=100).text)
```

- Input: `resource_id=` / `resource_ids=` for resources already on Datamint. `file_path` / `file_paths` refer to files in the server's approved storage directories, not the user's machine. For a local file, upload it first (`datamint-upload` skill) or predict locally.
- Version: same rules as deploy. Omit both to use the default `latest` image. `model_version=` uses the image deployed for that version. `model_alias=` uses the image built from that alias. The image must exist (deploy first), otherwise the server returns "No deployed image found".
- `save_results=True` writes predictions back to the resources as annotations. `save_results_options={'worklist_id': ..., 'annotation_source': ..., 'imported_from': ..., 'author_email': ...}` controls how they're saved, e.g. linking them to a worklist so annotators can review them.
- `params={...}` is forwarded to the model's `predict()`.
- `prompts={'text': ..., 'points': [{'label': 1, 'x': 120, 'y': 80}], 'boxes': [{'x_min': ..., 'y_min': ..., 'x_max': ..., 'y_max': ...}]}` for promptable segmentation models. Models without prompt support ignore them.
- `wait()` doesn't raise when the job fails. Check `inf.status` and `inf.error_message`.
- `inf.result_data` holds the raw result, and `inf.annotation_ids` holds the saved annotation ids when `save_results=True`.

Mode-specific helpers (same version/save/params options):

```python
api.inference.predict_image('unetpp_r34', resource_id=r.id)
api.inference.predict_frame('unetpp_r34', resource_id=video.id, frame_index=10)
api.inference.predict_slice('unetpp_r34', resource_id=ct.id, slice_index=40, axis='axial')
api.inference.predict_volume('unetpp_r34', resource_id=ct.id)
```

A model that implements only 2D prediction can still serve frames, slices and volumes: the SDK runs it per frame or slice. Per-slice results are 2D annotations, not a merged 3D segmentation.

## Serving-pod logs

```python
logs = api.pod_logs.get_logs('unetpp_r34', tag='latest', tail=500)   # server max 5000 lines
print(logs.pod_name, logs.status)
print(logs.text)

for pod in api.pod_logs.list_pods('unetpp_r34', tag='latest'):
    print(pod.name, pod.status, pod.is_running)

for line in api.pod_logs.stream_logs('unetpp_r34', tag='latest', interval=1.0):
    print(line, end='')
```

- Always pass `tag=`: the image tag of the deployment (`'latest'` for a default deploy, the alias name, or the version's tag). The `api.pod_logs` methods default to `tag='champion'`, which doesn't match a default deploy.
- `inf.get_pod_logs()` defaults to `tag='latest'`.
- Logs exist only while the pod exists (stopped pods included).

## Running a registered model locally

```python
version = api.models.get_by_name('unetpp_r34').get_latest_version()   # or get_latest_version(alias='prod')
model = version.load_model(device='cpu')
print(model.task_type, model.get_supported_modes())

preds = model.predict(resources)                                         # list of resources
preds = model.predict([ct], params={'mode': 'slice', 'slice_index': 4, 'axis': 'axial'})
preds = model.predict([video], params={'mode': 'frame', 'frame_index': 0})
```

The model registry is scoped to the active project: call `datamint.mlflow.set_project('MyProject')` first if the model isn't found. `LocalResource(local_filepath='scan.nii.gz')` wraps a local file as input.

CLI:

```bash
datamint inference file.png --model-name MyModel
datamint inference file.png --model-name MyModel --project MyProject --output result.png
datamint inference file.png --model-name MyModel --uncertainty
```

The project defaults to the model name. `--uncertainty` adds a predictive-entropy score per prediction (0 = confident, 1 = maximally uncertain). It's a cheap proxy, not a calibrated estimate. In Python: `model.predict(resources, params={'compute_uncertainty': True})`.

## Testing the serving image locally (podman)

Same image format and `/invocations` contract as the server, built and run on the user's machine. Requires podman.

```python
from datamint.mlflow.models.docker_build import build_docker_image, run_docker_container, stop_docker_container, remove_docker_image
from datamint.mlflow.models.local_inference import predict_local

built = build_docker_image('models:/unetpp_r34/latest', with_gpu=False)
running = run_docker_container(built.image_name, built.image_tag)      # waits until /ping responds
anns = predict_local(container_port=running.host_port, resource_id=r.id, api_client=api)   # or file_path= for a local file
stop_docker_container(running)
remove_docker_image(built.image_name, built.image_tag)
```

`models:/<name>/latest` is MLflow's URI for the highest version. `models:/<name>/3` and `models:/<name>@<alias>` select a version or alias.
