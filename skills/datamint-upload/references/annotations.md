# Annotating resources already on Datamint

Read this when the resources are already uploaded and the user wants to add labels, shapes or masks to them, or manage the worklist annotators use. For masks uploaded together with the resources, `SKILL.md` covers it.

## Before annotating

1. **Get a worklist id and pass it to every call (`worklist_id=`).** Annotations count toward a project only through one of its worklists, and the worklist can't be set afterwards.
   ```python
   proj = api.projects.get_by_name('Liver CT')
   worklists = proj.fetch_worklists()          # or api.annotationworklists.get_by_project(proj)
   ```
   `api.annotationworklists.get_list()` / `get_all()` raise `NotImplementedError`: there's no endpoint listing worklists across projects.
2. **The identifier must exist in the project or worklist setup.** Category values come from the list set up for that identifier. Box/line shapes are set up in the web app (Project Settings > Shapes and regions). Annotators only see the categories and segments listed in the worklist's `annotations` / segmentation schema.

## Creating a worklist

```python
worklist_id = api.annotationworklists.create(
    'Fracture review',
    resource_ids=[r.id for r in resources],
    project=proj,
    annotations=[{'type': 'category', 'identifier': 'fracture', 'scope': 'image',
                  'required': False, 'values': ['no', 'yes']}],
    annotators=[{'email': 'dr.silva@example.com'}],
    return_entity=False,
    exists_ok=True,
)
```

Annotation spec keys: `type`, `identifier`, `scope`, `required`, optional `values`.

For segmentation worklists, build `annotations` + `segmentation_data` from segment names with the static helper and pass both to `create`:

```python
schema = api.annotationworklists.segment_schema(['Liver', 'Tumor'])
api.annotationworklists.create('Liver review', resource_ids=ids, project=proj, **schema)
```

Updating an existing worklist (keeps what is already set):

```python
api.annotationworklists.add_segments(worklist, ['Spleen'])
api.annotationworklists.add_ai_segmentations(worklist, ['Liver', 'Tumor'], editable=True)
```

AI segmentations must be viewable to be shown at all. `add_ai_segmentations(..., editable=True)` adds them to both lists. When passing `editable_ai_segmentations` to `create` directly, include the same identifiers in `viewable_ai_segmentations`.

## Classification annotations

```python
api.annotations.add_category_annotation(res, identifier='position', value='Displaced', worklist_id=wl_id)
api.annotations.add_label_annotation(res, identifier='Fracture', worklist_id=wl_id)        # presence, no value
api.annotations.add_text_annotation(res, identifier='findings', value='small fracture line', worklist_id=wl_id)
api.annotations.create_numeric_annotation(res, identifier='diameter', value=12.5, units='mm', worklist_id=wl_id)
```

Frames (video, multi-frame DICOM): `frame_index=3` for one frame (0-based), `frame_range=(42, 46)` for frames 42 to 45 (end excluded, like `range()`). Without either, the annotation covers the whole resource.

`set_frame_ranges` sets the whole timeline of one identifier and **replaces** what was there:

```python
api.annotations.set_frame_ranges(res, identifier='position', type='category',
                                 values={'Displaced': [(0, 10), (20, 30)], 'Normal': [(10, 20)]},
                                 worklist_id=wl_id)
api.annotations.get_frame_ranges(res)
```

Use `None` as the value key for labels.

## Geometry annotations

Points come first, then `resource=` as a keyword:

```python
api.annotations.add_box_annotation((5, 5), (25, 20), resource=res, identifier='Lesion', frame_index=0, worklist_id=wl_id)
api.annotations.add_line_annotation((0, 0), (10, 30), resource=res, identifier='Length', frame_index=0, worklist_id=wl_id)
api.annotations.add_point_annotation((12, 40), resource=res, identifier='Landmark', frame_index=0, worklist_id=wl_id)
api.annotations.add_polyline_annotation([(0, 0), (10, 0), (10, 10)], resource=res, identifier='Contour',
                                        closed=True, frame_index=0, worklist_id=wl_id)
```

- Pixel coordinates are `(x, y)`.
- `coords_system='patient'` takes 3D patient coordinates and needs `metadata=` (a `pydicom.Dataset` or `Nifti1Image`).
- For volumes, `slice_plane` sets the view plane of `frame_index` (`from medimgkit import ViewPlane`, e.g. `ViewPlane.AXIAL`).
- Closed shapes (regions) use `add_polyline_annotation(..., closed=True)`.

## Segmentation masks

| Data | Method |
|---|---|
| 2D image, frames, video (PNG, numpy) | `api.annotations.upload_segmentations(res, path_or_array, name, worklist_id=...)` |
| 3D volume (NIfTI, 3D numpy array) | `api.annotations.upload_volume_segmentation(res, path_or_array, {1: 'Liver'}, worklist_id=...)` |

- `upload_segmentations` raises `ValueError` for NIfTI files. Use `upload_volume_segmentation`.
- `upload_segmentations` numpy shapes: `(H, W)` or `(H, W, frames)` grayscale, `(3, H, W, frames)` RGB. `name` is a string, `{pixel_value: name}`, or `{(r, g, b): name}`.
- `frame_index` maps mask frames to resource frames. Default: sequential from 0.
- `discard_empty_segmentations=True` (default) skips empty frames.
- `model_name=` marks them as AI segmentations from that model. It must be a registered model name.
- Make sure the segment names are in the worklist schema (`add_segments`), or annotators won't see them.

## Reading back

```python
for ann in api.annotations.get_list(resource=res):
    print(ann.identifier, ann.type)
```
