---
name: datamint-data-access
description: Explore, filter, load and download data from Datamint with the `datamint` Python SDK or the `datamint download` CLI. Covers listing projects, filtering resources (by project, tags, channel, modality, status, filename, date), loading images/DICOM/NIfTI/video into Python, slices and frames of volumes and videos, local caching, bulk downloads, reading annotations and segmentation masks, exporting annotations to CSV/Excel, and project annotation statistics. Use this whenever the user wants to see what's in a Datamint project, find specific scans, pull files or masks to local disk, inspect annotations, or check annotation progress, even if they don't say "download".
---

# Exploring and downloading Datamint data

```python
from datamint import Api
api = Api()  # credentials from `datamint config` or DATAMINT_API_KEY / DATAMINT_API_URL
```

For training, `ImageDataset` / `VolumeDataset` already download and cache everything (see the `datamint-training` skill). Use this skill for inspection, ad hoc analysis and exporting files.

## Projects

```python
projects = api.projects.get_all()
proj = api.projects.get_by_name('Liver CT')   # returns None if not found, doesn't raise
resources = proj.fetch_resources()            # always fresh from the server
worklists = proj.fetch_worklists()
```

`datamint.select_project('Liver CT')` sets a session default. Project methods with an optional `project=` argument (stats, exports) fall back to it.

## Finding resources

```python
res = api.resources.get_list(project_name='Liver CT', tags=['batch-1'], modality='CT', limit=50)
r = api.resources.get_by_id(resource_id)
```

`get_list` filters (all optional, combined with AND):

| Filter | Notes |
|---|---|
| `project_name` | Project name, `Project`, or list of names |
| `tags` | List of tags |
| `channel` | Upload channel name |
| `status` | `'inbox'` or `'published'`. `None` returns both |
| `modality`, `mimetype` | e.g. `'CT'`, `'application/dicom'` |
| `filename`, `source_filepath` | Exact filename / original path |
| `from_date`, `to_date` | Upload date, `to_date` exclusive |
| `order_field`, `order_ascending` | Sorting |
| `deleted` | `'exclude'` (default), `'include'`, `'only'` |
| `limit` | Max results |

Other ways to find resources:
- `api.resources.get_not_annotated(limit=...)`: resources with no annotations.
- `api.resources.list_channels()`: upload channels.
- `api.resources.rank_resources(resources, score_fn, top_k=10)`: sort by a custom score. `score_fn` returning `None` excludes a resource.

Useful resource attributes and checks: `id`, `filename`, `mimetype`, `modality`, `status`, `tags`, `size_mb`, `is_dicom()`, `is_nifti()`, `is_video()`, `is_volume()`, `is_multiframe()`, `get_patient_id()`, `show()` (opens it in the web app).

**List results can be partial.** Fields not returned by the list endpoint are fetched with an extra request the first time they're accessed. Accessing such a field in a loop over thousands of resources makes thousands of requests. If that's slow, read only what's needed.

## Loading file data into Python

```python
data = r.fetch_file_data(use_cache=True)            # auto_convert=True by default
raw = r.fetch_file_data(auto_convert=False)         # bytes
r.fetch_file_data(save_path='local/case1.dcm')      # also writes to disk
```

`auto_convert=True` returns:

| File type | Returned object |
|---|---|
| DICOM | `pydicom.Dataset` |
| NIfTI | `nibabel.Nifti1Image` |
| Image (PNG, JPEG, ...) | `PIL.Image.Image` |
| Video | `av` input container. The caller must close it (`with` block or `.close()`) |
| JSON | `dict` |

**Caching is opt-in.** `use_cache` defaults to `False`. Pass `use_cache=True` to read from and save to the local cache (`~/.datamint`), which revalidates against the server. `use_cache='loadonly'` reads the cache without saving misses.

Volumes and videos without handling the whole file:

```python
from medimgkit import ViewPlane
sl = r.get_slice(ViewPlane.AXIAL, 40)     # numpy array
frame = r.get_frame(10)                   # numpy array (video)
img = api.resources.download_resource_frame(r, frame_index=10)  # PIL image
```

`r.iter_slices(ViewPlane.AXIAL)` and `r.iter_frames()` return one proxy resource per slice or frame.

## Downloading to disk

```python
api.resources.download_resource_file(r, save_path='out/case1.dcm')
data, path = api.resources.download_resource_file(r, save_path='out/case1', add_extension=True)   # returns a tuple
paths = api.resources.download_multiple_resources(resources, save_path='out_dir', add_extension=True)
```

- `download_multiple_resources` is parallel and much faster than a loop. It takes a list (a single id raises `ValueError`), and `save_path` is a directory or a list of paths with the same length.
- `add_extension=True` requires `save_path` and appends the right extension. The returned paths are the real ones.

Pre-cache a whole project before heavy local work:

```python
proj.cache_resources(progress_bar=True)           # or api.resources.cache_resources(resources)
```

CLI equivalent, which also prints the matching dataset one-liner:

```bash
datamint download --project "Liver CT"
datamint download --project "Liver CT" -o ./liver_ct   # browsable folder of symlinks into the cache
```

Cache management: `datamint config --list-local-data`, `datamint config --clean-local-data resources`, `datamint config --clean-all-local-data`.

## Reading annotations

```python
anns = api.annotations.get_list(resource=r)
per_res = api.annotations.get_list(resource=resources, group_by_resource=True)   # one list per resource, same order
segs = api.annotations.get_list(resource=resources, annotation_type='segmentation', load_ai_segmentations=True)
anns = r.fetch_annotations(annotation_type='category')
```

`get_list` filters: `resource` (one or many, one request for many), `annotation_type`, `annotator_email`, `worklist_id`, `status` (`'new'` / `'published'`), `source` (`'manual'`, `'imported'`, `'model_pipeline'`, `'model_deploy'`), `from_date`, `to_date`, `deleted`, `limit`. Pass `load_ai_segmentations=True` when AI segmentations are needed.

Annotation attributes: `id`, `identifier`, `type`, `scope`, `frame_index`.

Segmentation masks:

```python
mask = seg.fetch_file_data(use_cache=True)                     # numpy array
results = api.annotations.download_multiple_files(segs, save_paths)
failed = [x for x in results if not x['success']]              # each dict has 'success', 'annotation_id', 'error'
```

## Exports and progress

```python
api.projects.download_annotations('annotations.csv', format='csv', project=proj)   # or 'xlsx'
api.projects.get_annotations_stats(proj)
api.projects.get_annotators_stats(proj)
api.projects.get_annotation_statuses(proj)
```

`download_annotations` filters: `annotators` (emails), `annotations` (identifiers), `from_date`, `to_date` (ISO strings).
