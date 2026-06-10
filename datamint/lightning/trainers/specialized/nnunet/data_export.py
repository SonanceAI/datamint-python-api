from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
import nibabel as nib
import numpy as np

_LOGGER = logging.getLogger(__name__)

NNUNET_SUFFIX = '_0000.nii.gz'


class DatamintToNNUNetExporter:
    """Exports a Datamint project's resources and annotations to nnUNet Task format.

    Directory layout produced under ``work_dir``:
    ::

        Dataset{id:03d}_{name}/
          imagesTr/case_001_0000.nii.gz
          labelsTr/case_001.nii.gz
          imagesTs/case_003_0000.nii.gz   (if test split present)
          dataset.json
          datamint_case_map.json

    Args:
        work_dir: Root directory (becomes ``nnUNet_raw`` in practice).
        dataset_id: 3-digit nnUNet dataset ID (e.g. 1 → ``001``).
        dataset_name: Human-readable name appended to the dataset dir (e.g. ``CTLiver``).
    """

    def __init__(self, work_dir: Path | str, dataset_id: int, dataset_name: str) -> None:
        self.work_dir = Path(work_dir)
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.dataset_dir = self.work_dir / f'Dataset{dataset_id:03d}_{dataset_name}'

    def _export_image(self, resource, case_id: str, split: str) -> Path:
        """Write one resource's NIfTI file to the nnUNet images directory.

        Reads the cached NIfTI directly via ``fetch_file_data`` — bypasses any
        albumentations transforms so voxel spacing and affine are preserved.

        Args:
            resource: Datamint resource (NIfTI or DICOM).
            case_id: nnUNet case identifier, e.g. ``'case_001'``.
            split: ``'train'`` writes to ``imagesTr/``; anything else to ``imagesTs/``.

        Returns:
            Path to the written ``.nii.gz`` file.
        """
        nifti_data = resource.fetch_file_data(use_cache=True, auto_convert=True)
        # fetch_file_data builds the Nifti1Image from a BytesIO / gzip stream and
        # calls get_fdata() to populate nibabel's internal cache before closing the
        # stream.  Reading via dataobj (ArrayProxy) would try to re-open the now-closed
        # stream — use get_fdata() to hit the cache instead.
        arr = nifti_data.get_fdata()

        out_dir = self.dataset_dir / ('imagesTr' if split == 'train' else 'imagesTs')
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f'{case_id}{NNUNET_SUFFIX}'
        nib.save(nib.Nifti1Image(arr, nifti_data.affine, nifti_data.header), str(out_path))
        _LOGGER.debug("Exported image %s → %s", resource.id, out_path)
        return out_path

    def _merge_segmentations(
        self,
        segs,
        name_to_idx: 'dict[str, int] | None' = None,
    ) -> np.ndarray:
        """Merge N segmentation masks into one int32 label map.

        Datamint returns annotations as per-class binary masks (0/1) where the
        class name is stored in ``ann.identifier``.  When ``name_to_idx`` is
        provided, each binary mask is scaled by its class index so the merged
        output contains the correct integer labels (e.g., aorta=1, liver=6).

        If an annotation already contains class-valued data (values > 1), it is
        used as-is so multi-class NIfTIs uploaded directly are also handled.

        When two masks assign different non-zero labels to the same voxel, the
        higher class index wins.  A ``UserWarning`` is raised whenever any
        overlap is found so the caller can inspect whether it is intentional.

        Args:
            segs: Sequence of annotation objects.
            name_to_idx: Mapping ``{class_name: class_index}`` used to scale
                binary masks to their correct integer class value.  When
                ``None`` the raw data is used unchanged.

        Returns:
            Merged ``np.ndarray`` of shape ``(H, W, D)`` and dtype ``int32``.
        """
        arrays = []
        for seg in segs:
            data = seg.fetch_file_data(auto_convert=True, use_cache=True)
            if isinstance(data, np.ndarray):
                arr = data.astype(np.int32)
            else:
                arr = data.get_fdata().astype(np.int32)

            # Binary mask (values in {0, 1}): scale by the class index so the
            # merged output encodes the correct integer label value.
            if name_to_idx is not None and set(np.unique(arr)).issubset({0, 1}):
                identifier = getattr(seg, 'identifier', None)
                if identifier and identifier in name_to_idx:
                    arr = arr * name_to_idx[identifier]

            arrays.append(arr)

        shape = arrays[0].shape
        merged = np.zeros(shape, dtype=np.int32)
        overlap_detected = False

        for seg_data in arrays:
            if not overlap_detected and np.any((merged > 0) & (seg_data > 0)):
                overlap_detected = True
            merged = np.maximum(merged, seg_data)

        if overlap_detected:
            warnings.warn(
                "Overlapping segmentations detected during merge. "
                "Highest class index wins at overlapping voxels.",
                UserWarning,
                stacklevel=2,
            )

        return merged

    def _export_label(
        self,
        resource,
        case_id: str,
        class_map: dict,
        annotations=None,
    ) -> Path:
        """Write the merged segmentation label map for one resource.

        Merges all segmentation annotations into a single int32 label map via
        :meth:`_merge_segmentations` and saves it under ``labelsTr/``.  The
        affine is taken from the resource's cached NIfTI so image and label
        share the same geometry.

        Args:
            resource: Datamint resource whose annotations to export.
            case_id: nnUNet case identifier, e.g. ``'case_001'``.
            class_map: Mapping of ``{int: class_name}`` used to scale binary
                masks to their correct integer label values.
            annotations: Pre-fetched annotation list.  When ``None`` the
                annotations are fetched from the server.

        Returns:
            Path to the written ``labelsTr/{case_id}.nii.gz`` file.
        """
        if annotations is None:
            annotations = resource.fetch_annotations(annotation_type='segmentation')

        if not annotations:
            raise ValueError(
                f"Resource '{resource.id}' has no segmentation annotations. "
                "All training resources must have at least one segmentation annotation "
                "for nnUNet export."
            )

        ref_nifti = resource.fetch_file_data(use_cache=True, auto_convert=True)
        # affine is stored in the header, not via ArrayProxy — safe to read directly.
        affine = ref_nifti.affine

        name_to_idx = {name: idx for idx, name in class_map.items()}
        merged = self._merge_segmentations(annotations, name_to_idx)

        out_dir = self.dataset_dir / 'labelsTr'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{case_id}.nii.gz'
        nib.save(nib.Nifti1Image(merged, affine), str(out_path))
        _LOGGER.debug("Exported label %s → %s", resource.id, out_path)
        return out_path

    def _write_dataset_json(
        self,
        labels: dict[str, int],
        channel_names: dict[str, str],
        num_training: int,
    ) -> Path:
        """Write nnUNet's ``dataset.json`` for this dataset.

        Args:
            labels: Class name → integer label mapping, e.g.
                ``{'background': 0, 'liver': 1}``.  Note the direction:
                name-keyed, int-valued — the inverse of Datamint's ``class_map``.
            channel_names: Modality index (as string) → modality name, e.g.
                ``{'0': 'CT'}``.
            num_training: Number of training cases.

        Returns:
            Path to the written ``dataset.json`` file.
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        content = {
            'labels': labels,
            'channel_names': channel_names,
            'numTraining': num_training,
            'file_ending': '.nii.gz',
        }
        out_path = self.dataset_dir / 'dataset.json'
        out_path.write_text(json.dumps(content, indent=2))
        _LOGGER.debug("Wrote dataset.json → %s", out_path)
        return out_path

    def _write_case_map(self, mapping: dict[str, str]) -> Path:
        """Write the sidecar JSON that maps nnUNet case IDs back to Datamint resource UUIDs.

        nnUNet uses sequential integer case names (``case_001``, ``case_002``, …)
        that have no connection to Datamint's UUID-based resource IDs.  This file
        is read by the result importer to route each prediction back to the correct
        resource.

        Args:
            mapping: ``{case_id: resource_uuid}`` e.g.
                ``{'case_001': 'res-uuid-001', 'case_002': 'res-uuid-002'}``.

        Returns:
            Path to the written ``datamint_case_map.json`` file.
        """
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.dataset_dir / 'datamint_case_map.json'
        out_path.write_text(json.dumps(mapping, indent=2))
        _LOGGER.debug("Wrote case map (%d entries) → %s", len(mapping), out_path)
        return out_path

    def export(self, split: dict, channel_names: dict[str, str]) -> Path:
        """Export a full Datamint project split to nnUNet Task format.

        Assigns sequential case IDs (``case_001``, ``case_002``, …) across all
        splits in order: train first, then test.  Train resources get both image
        and label files; test resources get only images (nnUNet never expects
        ``labelsTs/``).

        Args:
            split: ``{'train': [resources…], 'test': [resources…]}``.
                The ``'test'`` key is optional.
            channel_names: Modality index → name, e.g. ``{'0': 'CT'}``.
                Passed verbatim to ``dataset.json``.

        Returns:
            Path to the dataset directory (``Dataset{id:03d}_{name}/``).
        """
        train_resources = split.get('train', [])
        test_resources = split.get('test', [])

        case_map: dict[str, str] = {}
        global_class_map: dict[int, str] = {}
        case_counter = 1

        # Pass 1 — fetch annotations once per training resource, build the
        # global class map, and cache to avoid a second API call in _export_label.
        train_annotations: dict[str, list] = {}
        for resource in train_resources:
            annotations = resource.fetch_annotations(annotation_type='segmentation')
            train_annotations[resource.id] = annotations
            for ann in annotations:
                # Prefer explicit class_map (int→name) if present.
                class_map_attr = getattr(ann, 'class_map', None)
                if class_map_attr:
                    global_class_map.update(class_map_attr)
                else:
                    # Fallback: Datamint returns per-class binary masks where
                    # the class name lives in ann.identifier.  Assign the next
                    # available integer index (1-based, background=0 is reserved).
                    identifier = getattr(ann, 'identifier', None)
                    if identifier and identifier not in global_class_map.values():
                        next_idx = max(global_class_map.keys(), default=0) + 1
                        global_class_map[next_idx] = identifier

        # Pass 2 — export images and labels using the complete class map.
        for resource in train_resources:
            case_id = f'case_{case_counter:03d}'
            case_counter += 1
            case_map[case_id] = resource.id

            self._export_image(resource, case_id, 'train')
            self._export_label(
                resource, case_id, global_class_map,
                annotations=train_annotations[resource.id],
            )

        for resource in test_resources:
            case_id = f'case_{case_counter:03d}'
            case_counter += 1
            case_map[case_id] = resource.id
            self._export_image(resource, case_id, 'test')

        # nnUNet dataset.json expects name→int (inverse of Datamint's int→name).
        labels: dict[str, int] = {'background': 0}
        labels.update({name: idx for idx, name in global_class_map.items()})

        self._write_dataset_json(labels, channel_names, num_training=len(train_resources))
        self._write_case_map(case_map)

        _LOGGER.info(
            "Export complete: %d train, %d test → %s",
            len(train_resources), len(test_resources), self.dataset_dir,
        )
        return self.dataset_dir
