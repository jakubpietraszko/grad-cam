import numpy as np

import torch

import tensorflow as tf

def get_cam(image: torch.Tensor, model: torch.nn.Module, target_layer: torch.nn.Module) -> np.ndarray:
    '''
    Function to calculate CAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer

    Returns:
        np.ndarray: CAM
    '''

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    image = image.unsqueeze(0) # Add batch dimension shape (1, 3, x, y)

    model.eval()
    output = model(image)

    model.zero_grad()
    output[0, output.argmax()].backward()

    model.train()

    h1.remove()
    h2.remove()

    activations = activations.cpu().detach().numpy()
    gradients = gradients.cpu().detach().numpy()

    weights = np.mean(activations, axis=(2, 3))

    cam = np.zeros(activations.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        cam += w * activations[0, i, :, :]

    cam = np.maximum(cam, 0)

    return cam


def get_gradcam(image: torch.Tensor, model: torch.nn.Module, target_layer: torch.nn.Module) -> np.ndarray:
    '''
    Function to calculate GradCAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer

    Returns:
        np.ndarray: GradCAM
    '''

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    image = image.unsqueeze(0) # Add batch dimension shape (1, 3, x, y)

    model.eval()
    output = model(image)

    model.zero_grad()
    output[0, output.argmax()].backward()

    model.train()

    h1.remove()
    h2.remove()

    activations = activations.cpu().detach().numpy()
    gradients = gradients.cpu().detach().numpy()

    weights = np.mean(gradients, axis=(2, 3))

    gradcam = np.zeros(activations.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        gradcam += w * activations[0, i, :, :]

    gradcam = np.maximum(gradcam, 0)

    return gradcam

def get_hirescam(image: torch.Tensor, model: torch.nn.Module, target_layer: torch.nn.Module) -> np.ndarray:
    '''
    Function to calculate Hi-ResCAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer

    Returns:
        np.ndarray: Hi-ResCAM
    '''

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    image = image.unsqueeze(0) # Add batch dimension shape (1, 3, x, y)

    model.eval()
    output = model(image)

    model.zero_grad()
    output[0, output.argmax()].backward()

    model.train()

    h1.remove()
    h2.remove()

    activations = activations.cpu().detach().numpy()
    gradients = gradients.cpu().detach().numpy()

    weights = gradients

    hirescam = np.zeros(activations.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        hirescam += w * activations[0, i, :, :]

    hirescam = np.maximum(hirescam, 0)

    return hirescam
