import colorsys
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from torch import Tensor
from torchvision.transforms import functional as F

if TYPE_CHECKING:
    from medimgkit import ViewPlane

    from datamint.entities.annotations import Annotation
    from datamint.entities.resource import Resource

_LOGGER = logging.getLogger(__name__)


def show(imgs: Sequence[Tensor | np.ndarray] | Tensor | np.ndarray,
         figsize: tuple[int, int] | None = None,
         normalize: bool = False,
         ax: Axes | Sequence[Axes] | None = None):
    """
    Show a list of images in a grid.
    Args:
        imgs (Sequence[Tensor | np.ndarray] | Tensor | np.ndarray): List of images to show.
            Each image should be a tensor of shape (C, H, W) and dtype uint8 or float.
        figsize (tuple[int, int], optional): Size of the figure. Defaults to None.
        normalize (bool, optional): Whether to normalize the images to [0, 1] range by min-max scaling.
        ax (Axes | Sequence[Axes], optional): Matplotlib axes to plot on. If None, new axes are created.
            If a single Axes is provided and multiple images are given, only the first image is shown.
    """

    if not isinstance(imgs, list) and not isinstance(imgs, tuple):
        imgs = [imgs]

    imgs = [img if isinstance(img, torch.Tensor) else torch.from_numpy(img) for img in imgs]

    if normalize:
        for i, img in enumerate(imgs):
            img = img.float()
            img = img - img.min()
            img = img / img.max()
            imgs[i] = img

    if ax is None:
        if figsize is not None:
            _fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
        else:
            _fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        axs = axs[0]
    else:
        if isinstance(ax, Axes):
            axs = [ax]
        else:
            axs = list(ax)
    
    for i, img in enumerate(imgs):
        if i >= len(axs):
            break
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i].imshow(np.asarray(img))
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
           (200, 128, 0), (128, 200, 0), (0, 128, 200), (200, 0, 128), (128, 0, 200), (0, 200, 128)]


def generate_color_palette(num_objects: int) -> list[tuple[int, int, int]]:
    """
    Generate a list of colors for segmentation masks.

    Args:
        num_objects (int): Number of objects to generate colors for.

    Returns:
        List of RGB colors.
    """
    if num_objects <= 0:
        raise ValueError("Number of objects must be greater than 0.")

    colors = _COLORS[:num_objects]
    if len(colors) == num_objects:
        return colors

    num_objects -= len(colors)

    # generate random colors
    for _ in range(num_objects):
        hue = np.random.rand()
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))

    return colors


@torch.inference_mode()
def draw_masks(
    image: Tensor | np.ndarray,
    masks: Tensor | np.ndarray,
    alpha: float = 0.5,
    colors: Sequence[str | tuple[int, int, int]] | str | tuple[int, int, int] | None = None,
) -> Tensor:
    """
    Draws segmentation masks on given RGB image.
    This is different from `torchvision.utils.draw_segmentation_masks` as overlapping masks are blended together correctly.
    The image values should be uint8 or float.

    Args:
        image (Tensor): Tensor of shape (3, H, W) or (H, W) and dtype uint8 or float.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (color or list of colors, optional): List containing the colors
            of the masks or single color for all masks. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    if masks.ndim == 4:
        _LOGGER.warning(f"In draw_masks: Expected masks to have shape (num_masks, H, W) or (H, W), but got {masks.shape}."
                        " It might produce unexpected results. Please check the shape of the masks.")

    if image.ndim == 3 and image.shape[0] == 1:
        # convert to RGB
        image = image.expand(3, -1, -1)
    if image.ndim == 2:
        # convert to RGB
        image = image.unsqueeze(0).expand(3, -1, -1)

    if masks.dtype != torch.bool:
        masks = masks.bool()

    # if image has negative values, scale to [0, 1]
    if image.min() < 0:
        image = image.float()
        image = image - image.min()
        image = image / image.max()

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    if colors is None:
        colors = generate_color_palette(len(masks))

    for color, mask in zip(colors, masks):
        image = torchvision.utils.draw_segmentation_masks(image=image,
                                                          masks=mask,
                                                          alpha=alpha,
                                                          colors=color)
    return image


# --- Resource plots ---

def _std_axis_idx(resource: 'Resource', metainfo, axis: 'ViewPlane') -> int:
    """Axis of the (N, C, H, W) array that matches the anatomical plane, same as SlicedVolumeResource."""
    from medimgkit import dicom_utils, nifti_utils

    if resource.is_dicom():
        return dicom_utils.rawplaneaxis2stdplaneaxis_idx(dicom_utils.get_plane_axis(metainfo, axis))
    if resource.is_nifti():
        return nifti_utils.rawplaneaxis2stdplaneaxis_idx(nifti_utils.get_plane_axis(metainfo, axis))
    raise ValueError(f'Unsupported resource type for slicing: {resource.filename}|{resource.mimetype}')


def _load_masks(anns: Sequence['Annotation'], depth: int) -> tuple[np.ndarray | None, list[str]]:
    """(M, N, H, W) binary masks, one per (author, identifier), and their legend names."""
    from medimgkit.readers import read_array_normalized

    groups: dict[tuple[str, str], np.ndarray] = {}
    for ann in anns:
        mask = read_array_normalized(ann.fetch_file_data(auto_convert=False, use_cache=True)).any(1)  # (N, H, W)
        key = (ann.created_by or getattr(ann, 'created_by_model', None) or 'unknown', ann.identifier)
        if ann.scope == 'frame':
            vol = groups.setdefault(key, np.zeros((depth, *mask.shape[1:]), dtype=bool))
            vol[ann.frame_index] |= mask[0]
        else:
            groups[key] = groups[key] | mask if key in groups else mask

    if not groups:
        return None, []
    multiple_authors = len({author for author, _ in groups}) > 1
    names = [f'{identifier} ({author})' if multiple_authors else identifier for author, identifier in groups]
    return np.stack(list(groups.values())), names


_SHAPE_TYPES = ('square', 'line', 'point', 'region', 'circle', 'spline', 'distance', 'angle', 'area')
_LABEL_TYPES = ('category', 'label', 'integer', 'float', 'text')


def _read_volume(resource: 'Resource', axis: 'ViewPlane | None'):
    """(N, C, H, W) array, its metainfo and the array axis to slice along."""
    from medimgkit.readers import read_array_normalized

    raw = resource.fetch_file_data(auto_convert=False, use_cache=True)
    vol, metainfo = read_array_normalized(raw, return_metainfo=True)
    std_axis = _std_axis_idx(resource, metainfo, axis) if resource.is_volume() else 0
    return vol, metainfo, std_axis


def _get_image(vol: np.ndarray, index: int, std_axis: int) -> np.ndarray:
    """(C, a, b) slice scaled to [0, 1]."""
    img = np.take(vol, index, axis=std_axis)  # (C, a, b), or (a, C, b) when slicing H or W
    if std_axis != 0:
        img = np.moveaxis(img, 1, 0)
    img = img.astype(np.float32)
    return np.ascontiguousarray((img - img.min()) / (img.max() - img.min() + 1e-8))


def _plot_title(resource: 'Resource', axis: 'ViewPlane | None', index: int) -> str:
    return f'{resource.filename} | {axis} {index}' if resource.is_volume() else f'{resource.filename} | {index}'


def plot_segmentations(resource: 'Resource',
                       index: int | None = None,
                       axis: 'ViewPlane' = 'axial',
                       alpha: float = 0.5,
                       ax: Axes | None = None) -> Axes:
    """Plot one slice (or frame, or the 2D image) of ``resource`` with all its segmentations on top.

    ``index=None`` picks the slice with the biggest mask. ``axis`` is only used for volumes.
    """
    vol, _, std_axis = _read_volume(resource, axis)
    seg_axis = 0 if std_axis == 0 else std_axis - 1  # masks are (N, H, W), without the channel axis

    anns = resource.fetch_annotations(annotation_type='segmentation')
    masks, names = _load_masks(anns, depth=vol.shape[0])
    if masks is None:
        _LOGGER.warning(f'No segmentations found for resource {resource.filename!r}')

    if index is None:
        if masks is None:
            index = 0
        else:
            area = masks.any(0).sum(axis=tuple(a for a in range(3) if a != seg_axis))
            index = int(area.argmax())

    img = _get_image(vol, index, std_axis)
    if ax is None:
        _fig, ax = plt.subplots()

    if masks is None:
        show(img, ax=ax)
    else:
        colors = generate_color_palette(len(masks))
        slice_masks = np.ascontiguousarray(np.take(masks, index, axis=seg_axis + 1))  # (M, a, b)
        show(draw_masks(img, slice_masks, alpha=alpha, colors=colors), ax=ax)
        ax.legend(handles=[Patch(color=np.array(c) / 255, label=n) for c, n in zip(colors, names)],
                  loc='upper right', fontsize='small')

    ax.set_title(_plot_title(resource, axis, index), fontsize='small')
    return ax


def _draw_shape(ax: Axes, ann_type: str, pts: np.ndarray, color, closed: bool = False) -> None:
    """``pts`` are (P, 2) pixel coordinates on the plotted slice."""
    if ann_type == 'point':
        ax.scatter(pts[:, 0], pts[:, 1], color=color, marker='x')
        return
    if ann_type == 'square':
        pts = pts[[0, 1, 3, 2]]  # corners are stored as (p1, top right, bottom left, p2)
        closed = True
    if closed:
        pts = np.vstack([pts, pts[:1]])
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5)


def plot_shapes(resource: 'Resource',
                axis: 'ViewPlane | None' = None,
                index: int | None = None,
                ax: Axes | None = None) -> Axes:
    """Plot one slice (or the 2D image) of ``resource`` with its boxes, lines, points and regions.

    ``axis`` is required for volumes: shapes are only shown on the plane they were drawn on.
    ``index=None`` picks the slice with the most shapes.
    """
    from datamint.entities.annotations.geometry import Geometry

    if resource.is_volume() and axis is None:
        raise ValueError("axis is required for volumes, e.g. axis='axial'")

    vol, metainfo, std_axis = _read_volume(resource, axis)
    plane = axis if resource.is_volume() else None

    anns = [ann for ann in resource.fetch_annotations() if ann.annotation_type in _SHAPE_TYPES]
    shapes = []  # (annotation, (P, 3) slice coordinates)
    skipped: dict[str, int] = {}
    for ann in anns:
        geom = ann.geometry
        if not isinstance(geom, Geometry):
            skipped[ann.annotation_type] = skipped.get(ann.annotation_type, 0) + 1
            continue
        pts = np.asarray(geom.to_pixel_coordinates(metainfo, plane).points, dtype=float)
        if np.ptp(pts[:, 2]) > 0.5:
            skipped['other plane'] = skipped.get('other plane', 0) + 1
            continue
        shapes.append((ann, pts))
    if skipped:
        _LOGGER.warning(f'Shapes not shown: {skipped}')

    slice_of = [int(round(pts[0, 2])) for _, pts in shapes]
    if index is None:
        index = max(set(slice_of), key=slice_of.count) if shapes else 0

    if ax is None:
        _fig, ax = plt.subplots()
    show(_get_image(vol, index, std_axis), ax=ax)

    names = sorted({ann.identifier for ann in anns})
    colors = {name: np.array(c) / 255 for name, c in zip(names, generate_color_palette(len(names)))} if names else {}
    for (ann, pts), k in zip(shapes, slice_of):
        if k == index:
            _draw_shape(ax, ann.annotation_type, pts[:, :2], colors[ann.identifier],
                        closed=getattr(ann.geometry, 'closed', False))
    if colors:
        ax.legend(handles=[Patch(color=c, label=n) for n, c in colors.items()], loc='upper right', fontsize='small')

    ax.set_title(_plot_title(resource, axis, index), fontsize='small')
    return ax


def plot_labels(resource: 'Resource',
                index: int | None = None,
                axis: 'ViewPlane' = 'axial',
                ax: Axes | None = None) -> Axes:
    """Plot one slice (or the 2D image) of ``resource`` with its categories, labels, numbers and texts listed on top.

    Frame labels are shown on the native slices only.
    ``index=None`` picks the first slice with a frame label, or the middle slice.
    """
    vol, _, std_axis = _read_volume(resource, axis)
    anns = [ann for ann in resource.fetch_annotations() if ann.annotation_type in _LABEL_TYPES]

    def on_slice(ann, k: int) -> bool:
        if ann.scope != 'frame':
            return True
        if std_axis != 0:
            return False
        if ann.frame_ranges:
            return any(start <= k <= end for start, end in ann.frame_ranges)
        return ann.frame_index == k

    if index is None:
        frames = sorted(ann.frame_index for ann in anns if ann.scope == 'frame' and ann.frame_index is not None)
        index = frames[0] if frames and std_axis == 0 else vol.shape[std_axis] // 2

    lines = []
    for ann in anns:
        if not on_slice(ann, index):
            continue
        value = ann.text_value if ann.text_value is not None else ann.numeric_value
        if value is None:
            lines.append(ann.identifier)
        else:
            lines.append(f'{ann.identifier}: {value}' + (f' {ann.units}' if ann.units else ''))

    if ax is None:
        _fig, ax = plt.subplots()
    show(_get_image(vol, index, std_axis), ax=ax)
    ax.text(0.01, 0.99, '\n'.join(lines) if lines else 'no labels', transform=ax.transAxes,
            va='top', ha='left', fontsize='small', color='white',
            bbox={'facecolor': 'black', 'alpha': 0.6, 'edgecolor': 'none'})

    ax.set_title(_plot_title(resource, axis, index), fontsize='small')
    return ax
