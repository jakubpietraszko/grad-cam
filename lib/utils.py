import PIL
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision.transforms

import tensorflow as tf

def get_image_torch(path: Path, x: int, y: int) -> torch.Tensor:
    '''
    Function to read image from path and resize it to x, y

    Args:
        path (Path): Path to image
        x (int): Width of image
        y (int): Height of image

    Returns:
        torch.Tensor: Image tensor with shape (3, x, y)
    '''
    img = PIL.Image.open(path)
    img = img.resize((x, y))
    img = torchvision.transforms.ToTensor()(img)
    return img

def get_image_tf(path: Path, x: int, y: int) -> tf.Tensor:
    '''
    Function to read image from path and resize it to x, y

    Args:
        path (Path): Path to image
        x (int): Width of image
        y (int): Height of image

    Returns:
        tf.Tensor: Image tensor with shape (x, y, 3)
    '''
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, (x, y))
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


def overlay_plot_torch(image: torch.Tensor, cam: np.ndarray, alpha: float, save_path: Path = None) -> None:
    '''
    Function to overlay CAM on image
    It shows the image, overlayed image, and CAM.

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        cam (np.ndarray): CAM with shape (x_, y_) x_ and y_ smaller than x and y
        alpha (float): Alpha value for overlay

    Returns:
        None
    '''
    img = image.permute(1, 2, 0).numpy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam_resized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    overlayed = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].axis('off')

    ax[1].imshow(overlayed)
    ax[1].axis('off')

    ax[2].imshow(cam_resized, cmap='jet')
    ax[2].axis('off')

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_torch(image: torch.Tensor, save_path: Path = None) -> None:
    '''
    Function to plot image

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)

    Returns:
        None
    '''
    img = image.permute(1, 2, 0).numpy()

    plt.imshow(img)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)

    plt.show()

def delete_the_least_important_pixels(image: torch.Tensor, cam: np.ndarray, percentage: float) -> torch.Tensor:
    '''
    Function to delete the least important pixels from the image
    It uses CAM to determine which pixels are the least important

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        cam (np.ndarray): CAM with shape (x_, y_) x_ and y_ smaller than x and y
        percentage (float): Percentage of pixels to delete

    Returns:
        torch.Tensor: Image tensor with shape (3, x, y)
    '''

    cam_resized = cv2.resize(cam, (image.shape[2], image.shape[1]))
    cam_resized = cv2.normalize(cam_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask = cam_resized <= int(percentage * 255)

    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np[mask] = [0, 0, 0]

    return torch.from_numpy(img_np).permute(2, 0, 1).float()