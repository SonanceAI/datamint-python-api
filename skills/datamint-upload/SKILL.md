---
name: datamint-upload
description: Upload data to Datamint with the `datamint` Python SDK or the `datamint upload` / `datamint import` CLI. Covers DICOM files and series, NIfTI volumes, images and videos, with projects, tags, channels, metadata, anonymization, segmentation masks and AI pre-annotations for review in a worklist. Also covers adding annotations (categories, labels, text, boxes, lines, polylines, segmentations) to resources already on Datamint. Use this whenever the user wants to get files or labels into Datamint, import a COCO/YOLO/Pascal VOC dataset, attach model predictions for annotators to correct, or asks why uploaded annotations or masks don't show up in a project or worklist.
---

# Uploading data to Datamint

```python
from datamint import Api
api = Api()  # credentials from `datamint config` or DATAMINT_API_KEY / DATAMINT_API_URL
```

## Pick the entry point

| Situation | Use |
|---|---|
| One file | `api.resources.upload_resource(path, ...)` |
| Several files (any mix of types) | `api.resources.upload_resources([paths], ...)` |
| A folder, from the terminal | `datamint upload <dir> -r ...` (see CLI below) |
| A labeled COCO / Pascal VOC / YOLO dataset | `datamint import <dir> --project P` |
| Annotations for resources already on Datamint | read `references/annotations.md` |

`upload_resources` requires a sequence. Passing a single path raises `ValueError`, use `upload_resource` for that.

## Rules that fail silently or late

- **Data that belongs to a project goes in with `publish_to=` in the same call.** Annotations count toward a project only through one of its worklists, and a worklist is set when an annotation is created. Masks uploaded without `publish_to` are attached to the resource but never link to a project worklist afterwards. Fixing that means re-uploading the annotations.
- **The project must exist.** `publish_to` with an unknown project raises `ItemNotFoundError`. Create it first with `api.projects.create(name, description, exists_ok=True)` (`description` is required).
- **Anonymization differs between SDK and CLI.** The SDK default is `anonymize=False`. The CLI anonymizes DICOMs by default (`--retain-pii` turns it off). Set `anonymize=True` explicitly when uploading patient DICOMs from Python.
- **Masks and multi-file DICOM series don't mix.** `assemble_dicoms=True` (default) groups DICOM files of the same series into one resource. If that grouping happens and `segmentation_files` is given, the upload raises `NotImplementedError`. Upload masks with single-file resources (NIfTI volume, multi-frame DICOM, image, video), or upload the series first and attach masks afterwards (`references/annotations.md`).
- **Name the segments.** Segment names come from `segmentation_files[i]['names']`. Without names the worklist doesn't know which segments to show, and the SDK logs a warning that masks may not appear.
- **`on_error='skip'` returns exceptions inside the result list.** Check `isinstance(r, Exception)` before using the ids.

## `upload_resources` parameters

| Parameter | Notes |
|---|---|
| `files_path` | Paths, file-like objects or `pydicom.Dataset`, mixed types allowed |
| `publish_to` | Project instance, name or id. Resources get status `published` and are added to the project. `publish` is ignored when set |
| `publish` | `True` publishes without a project. Default leaves resources in the `inbox` |
| `tags` | List of strings, used later for filtering (`get_list(tags=[...])`) |
| `channel` | Free-form name to group resources |
| `anonymize`, `anonymize_retain_codes` | DICOM anonymization, and DICOM tags to keep, e.g. `[(0x0008, 0x0050)]` |
| `assemble_dicoms` | Default `True`: one resource per DICOM series. The returned list still has one entry per input file |
| `discard_dicom_reports` | Default `True`: skips DICOM structured reports |
| `metadata` | One entry per file: dict, path to a JSON file, or `None` |
| `modality`, `mimetype` | Optional, guessed when omitted |
| `mung_filename` | Keep folder parts in the stored filename: `'all'` or a list of depths |
| `on_error` | `'raise'` (default) or `'skip'` |
| `progress_bar` | Show a progress bar |
| `segmentation_files` | Masks per resource, see next section |
| `model_name`, `worklist`, `ai_segmentations`, `transpose_segmentation` | See next section |

Returns `list[str | Exception]`, one entry per input file, in input order.

## Masks at upload time

`segmentation_files` has one entry per file in `files_path`. Use `None` for files without masks.

```python
segmentation_files = [
    {'files': ['case1_seg.nii.gz'], 'names': {1: 'Liver', 2: 'Tumor'}},  # pixel value -> segment name
    None,                                                                 # no mask for this file
]
```

- `files`: list of mask files for that resource. The mask must match the image geometry (same size and frames/slices). If axes are swapped, pass `transpose_segmentation=True`.
- `names`: `{pixel_value: name}` dict, or a list of names with the same length as `files`.

With `publish_to` set:
- Resources are added to the worklist `'Imported annotations'` of the project (created if missing), and the masks are linked to it. Resources without masks are added too, so they show as still to annotate.
- `worklist=` picks another worklist by instance, id or name. A name not found in the project creates a new worklist. An id not found raises `ItemNotFoundError`. `worklist` without `publish_to` raises `ValueError`.
- The worklist's segment schema is built from the segment names.

### AI pre-annotations for human review

Masks produced by a model become AI segmentations when `model_name` is given. `model_name` must be a model registered in the Datamint model registry (`api.models.get_by_name(name)` returns it). An unknown name raises `ItemNotFoundError` listing the available models.

- `ai_segmentations='editable'` (default): annotators start from the AI mask and save their own corrected annotation.
- `ai_segmentations='viewable'`: read-only.
- AI segmentations alone don't count a file as annotated. It counts once an annotator saves an annotation or marks it done.

```python
proj = api.projects.create('Liver CT', 'CT scans with liver/tumor masks', exists_ok=True)

ids = api.resources.upload_resources(
    ['ct/case1.nii.gz', 'ct/case2.nii.gz'],
    publish_to=proj,
    tags=['liver-ct', 'batch-1'],
    segmentation_files=[
        {'files': ['pred/case1.nii.gz'], 'names': {1: 'Liver', 2: 'Tumor'}},
        {'files': ['pred/case2.nii.gz'], 'names': {1: 'Liver', 2: 'Tumor'}},
    ],
    model_name='liver-seg-v2',         # omit for human-made masks
    worklist='Liver review',           # omit to use 'Imported annotations'
    on_error='skip',
    progress_bar=True,
)
failed = [(p, r) for p, r in zip(['ct/case1.nii.gz', 'ct/case2.nii.gz'], ids) if isinstance(r, Exception)]
```

## CLI

```bash
datamint upload ./dicoms -r --project "Liver CT" --tag batch-1
datamint upload ./dicoms -r --segmentation_path ./segs --segmentation_names names.yaml --project "Liver CT"
datamint upload ./dicoms -r --segmentation_path ./segs --segmentation_names names.yaml \
    --project "Liver CT" --ai-model liver-seg-v2 --worklist "Liver review" --yes
```

- DICOMs are anonymized by default. `--retain-pii` keeps PII, `--retain-attribute (0008, 0050)` keeps one tag.
- `--segmentation_path` must mirror the folder structure of the resources.
- `--segmentation_names` takes a YAML file with `segmentation_names` (matched against mask filenames) and optional `class_names` (`{pixel_value: name}`), or an ITK-SNAP label export (CSV/TXT).
- `--include-extensions dcm` / `--exclude-extensions txt csv` filter files. Common non-medical extensions are excluded by default.
- NIfTI JSON sidecars with the same base name are included as metadata by default (`--no-auto-detect-json` disables).
- `--publish` publishes without a project. `--channel`, `--mungfilename`, `--no-assemble-dicoms`, `--transpose-segmentation` mirror the SDK options.
- The command prints a summary and asks for confirmation. Pass `--yes` when running non-interactively.

### Labeled datasets

```bash
datamint import ./my_dataset --project MyProject --dry-run   # parse and print counts only
datamint import ./my_dataset --project MyProject
datamint import ./my_dataset --project MyProject --format yolo --images-dir ./images --labels-dir ./labels
```

Imports images plus bounding-box annotations from COCO, Pascal VOC or YOLO. Format and layout are auto-detected and confirmed before upload.

## Verify the result

```python
res = api.resources.get_list(project_name='Liver CT', tags=['batch-1'])
for wl in proj.fetch_worklists():
    print(wl.id, wl.name)
```

If masks don't show in a worklist, check in this order: was `publish_to` passed in the upload call, were segment names given, and for AI masks, is the model name a registered model.
