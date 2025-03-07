import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as F
from torch import Tensor
import torchvision.utils
import torch
import colorsys


def show(imgs: list[Tensor] | Tensor,
         figsize: tuple[int, int] = None,
         normalize: bool = False):
    """
    Show a list of images in a grid.
    Args:
        imgs (list[Tensor] | Tensor): List of images to show.
            Each image should be a tensor of shape (C, H, W) and dtype uint8 or float.
        figsize (tuple[int, int], optional): Size of the figure. Defaults to None.
        normalize (bool, optional): Whether to normalize the images to [0, 1] range by min-max scaling.
    """

    if not isinstance(imgs, list):
        imgs = [imgs]

    if normalize:
        for i, img in enumerate(imgs):
            img = img.float()
            img = img - img.min()
            img = img / img.max()
            imgs[i] = img

    if figsize is not None:
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    else:
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


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


@torch.no_grad()
def draw_masks(
    image: Tensor,
    masks: Tensor,
    alpha: float = 0.5,
    colors: list[str | tuple[int, int, int]] | str | tuple[int, int, int] | None = None,
) -> Tensor:
    """
    Draws segmentation masks on given RGB image.
    This is different from `torchvision.utils.draw_segmentation_masks` as overlapping masks are blended together correctly.
    The image values should be uint8 or float.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8 or float.
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
    

    if image.ndim == 3 and image.shape[0] == 1:
        # convert to RGB
        image = image.expand(3, -1, -1)

    if masks.dtype != torch.bool:
        masks = masks.bool()

    # if image has negative values, scale to [0, 1]
    if image.min() < 0:
        image = image.float()
        image = image - image.min()
        image = image / image.max()

    if masks.ndim == 2:
        return torchvision.utils.draw_segmentation_masks(image=image,
                                                         masks=masks,
                                                         alpha=alpha,
                                                         colors=colors)

    if colors is None:
        colors = generate_color_palette(len(masks))

    for color, mask in zip(colors, masks):
        image = torchvision.utils.draw_segmentation_masks(image=image,
                                                          masks=mask,
                                                          alpha=alpha,
                                                          colors=color)
    return image
