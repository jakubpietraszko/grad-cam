import PIL
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision.transforms


from PIL import Image
import torchvision.transforms as T
import torch


def get_image_torch(path: Path, x: int, y: int) -> torch.Tensor:
    """
    Function to read image from path and resize it to x, y

    Args:
        path (Path): Path to image
        x (int): Width of image
        y (int): Height of image

    Returns:
        torch.Tensor: Image tensor with shape (3, x, y)
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((x, y))
    img = torchvision.transforms.ToTensor()(img)
    return img


"""def overlay_plot(path: Path, cam: np.ndarray, alpha: float) -> None:
    '''
    Function to overlay CAM on image
    It showes image, overlayed image and CAM

    Args:
        path (Path): Path to image
        cam (np.ndarray): CAM
        alpha (float): Alpha value for overlay

    Returns:
        None
    '''

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_resized = np.uint8(255 * cam_resized / cam_resized.max())
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)   

    overlayed = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title('Image')
    ax[0].axis('off')

    ax[1].imshow(overlayed)
    ax[1].set_title('Overlayed')
    ax[1].axis('off')

    ax[2].imshow(cam_resized, cmap='jet')
    ax[2].set_title('CAM')
    ax[2].axis('off')

    plt.show()"""


def overlay_plot_torch(
    image: torch.Tensor, cam: np.ndarray, alpha: float, save_path: Path = None
) -> None:
    """
    Function to overlay CAM on image
    It shows the image, overlayed image, and CAM.

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        cam (np.ndarray): CAM with shape (x_, y_) x_ and y_ smaller than x and y
        alpha (float): Alpha value for overlay

    Returns:
        None
    """
    img = image.permute(1, 2, 0).numpy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_resized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Invert the heatmap values
    cam_resized = 255 - cam_resized

    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    overlayed = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].axis("off")

    ax[1].imshow(overlayed)
    ax[1].axis("off")

    cam_resized = 255 - cam_resized

    ax[2].imshow(cam_resized, cmap="jet")
    ax[2].axis("off")

    if save_path:
        plt.savefig(save_path)

    plt.show()


def get_overlay(image: torch.Tensor, cam: np.ndarray, alpha: float) -> torch.Tensor:
    """
    Function to return ready overlay image

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        cam (np.ndarray): CAM with shape (x_, y_) x_ and y_ smaller than x and y
        alpha (float): Alpha value for overlay

    Returns:
        torch.Tensor: Image tensor with shape (3, x, y)
    """

    img = image.permute(1, 2, 0).numpy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_resized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Invert the heatmap values
    cam_resized = 255 - cam_resized

    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    overlayed = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    cam_resized = 255 - cam_resized

    return overlayed


def plot_torch(image: torch.Tensor, save_path: Path = None) -> None:
    """
    Function to plot image

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)

    Returns:
        None
    """
    img = image.permute(1, 2, 0).numpy()

    plt.imshow(img)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)

    plt.show()


def delete_the_least_important_pixels(
    image: torch.Tensor, cam: np.ndarray, percentage: float
) -> torch.Tensor:
    """
    Function to delete the least important pixels from the image
    It uses CAM to determine which pixels are the least important

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        cam (np.ndarray): CAM with shape (x_, y_) x_ and y_ smaller than x and y
        percentage (float): Percentage of pixels to delete [0, 1]

    Returns:
        torch.Tensor: Image tensor with shape (3, x, y)
    """

    cam_resized = cv2.resize(cam, (image.shape[2], image.shape[1]))
    cam_resized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    mask = cam_resized <= int(percentage * 255)

    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np[mask] = [0, 0, 0]

    return torch.from_numpy(img_np).permute(2, 0, 1).float()


def average_drop(Y: np.ndarray, O: np.ndarray) -> float:
    """
    Function to calculate average drop in output values

    Args:
        Y (np.ndarray): Output of plain image
        O (np.ndarray): Output of image with deleted pixels

    Returns:
        float: Average drop in output values
    """
    return np.maximum(0, (Y - O)).sum() / Y.shape[0]


def rate_of_increase_in_score(Y: np.ndarray, O: np.ndarray) -> float:
    """
    Function to calculate rate of increase in score

    Args:
        Y (np.ndarray): Output of plain image
        O (np.ndarray): Output of image with deleted pixels

    Returns:
        float: Rate of increase in score
    """
    return (Y < O).sum() / Y.shape[0]


id_name = dict()
dups_id_name = dict()

for i, name in enumerate(open("imagenet_classes.txt", "r")):
    name = name[:-1]
    name = name.lower()
    name = name.replace(" ", "_")
    id_name[i] = name

len(id_name)
hash_name = dict()
dups_hash_name = dict()

for line in open("imagenet.txt", "r"):
    line = line[:-1]
    h, i, n = line.split()
    n = n.lower()
    hash_name[h] = n
len(hash_name)
DICT = dict()

for k1, v1 in id_name.items():
    for k2, v2 in hash_name.items():
        if v1 == v2:
            DICT[k1] = (k1, k2, v1)
            DICT[k2] = (k1, k2, v1)
len(DICT)
