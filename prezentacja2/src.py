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

    image = image.unsqueeze(0)

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

def get_ablationcam(image: torch.Tensor, model: torch.nn.Module, target_layer: torch.nn.Module) -> np.ndarray:
    '''
    Function to calculate AblationCAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer

    Returns:
        np.ndarray: AblationCAM
    '''

    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    h1 = target_layer.register_forward_hook(forward_hook)

    image = image.unsqueeze(0)  # Shape (1, 3, x, y)
    model.eval()


    pred = model(image)
    y_c = pred[0, pred.argmax()].item()

    h1.remove()

    activations = activations.cpu().detach().numpy()
    weights = np.zeros(activations.shape[1], dtype=np.float32)

    for i in range(activations.shape[1]):
        def hook_zeros_k_map(module, inp, out):
            modified_output = out.clone()
            modified_output[:, i] = 0
            return modified_output

        hook = target_layer.register_forward_hook(hook_zeros_k_map)

        pred_mod = model(image)
        y_k = pred_mod[0, pred.argmax()].item()

        weights[i] = (y_c - y_k) / y_c if y_c != 0 else 0

        hook.remove()

    h1.remove()

    activations = activations[0]
    ablationcam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        ablationcam += float(w) * activations[i, :, :]

    ablationcam = np.maximum(ablationcam, 0)

    return ablationcam

