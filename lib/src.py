import numpy as np

import torch

import cv2


import torch.nn.functional as F
from tqdm import tqdm


def get_gradcam(
    image: torch.Tensor,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    target_class: int = None,
) -> np.ndarray:
    """
    Function to calculate GradCAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer
        target_class (int | None): Target class

    Returns:
        np.ndarray: GradCAM
    """

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

    if target_class is None:
        output[0, output.argmax()].backward()
    else:
        output[0, target_class].backward()

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
    gradcam = cv2.normalize(gradcam, None, 0, 1, cv2.NORM_MINMAX)

    return gradcam


def get_hirescam(
    image: torch.Tensor,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    target_class: int = None,
) -> np.ndarray:
    """
    Function to calculate Hi-ResCAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer
        target_class (int | None): Target

    Returns:
        np.ndarray: Hi-ResCAM
    """

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

    image = image.unsqueeze(0)  # Add batch dimension shape (1, 3, x, y)

    model.eval()
    output = model(image)

    model.zero_grad()

    if target_class is None:
        output[0, output.argmax()].backward()
    else:
        output[0, target_class].backward()

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
    hirescam = cv2.normalize(hirescam, None, 0, 1, cv2.NORM_MINMAX)

    return hirescam


def get_ablationcam(
    image: torch.Tensor,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    target_class: int = None,
) -> np.ndarray:
    """
    Function to calculate AblationCAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer
        target_class (int | None): Target class

    Returns:
        np.ndarray: AblationCAM
    """

    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    h1 = target_layer.register_forward_hook(forward_hook)

    image = image.unsqueeze(0)  # Shape (1, 3, x, y)
    model.eval()

    pred = model(image)

    if target_class is None:
        y_c = pred[0, pred.argmax()].item()
    else:
        y_c = pred[0, target_class].item()

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

        if target_class is None:
            y_k = pred_mod[0, pred_mod.argmax()].item()
        else:
            y_k = pred_mod[0, target_class].item()

        weights[i] = (y_c - y_k) / y_c if y_c != 0 else 0

        hook.remove()

    h1.remove()

    activations = activations[0]
    ablationcam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        ablationcam += float(w) * activations[i, :, :]

    ablationcam = np.maximum(ablationcam, 0)
    ablationcam = cv2.normalize(ablationcam, None, 0, 1, cv2.NORM_MINMAX)

    return ablationcam


def get_gradcamplusplus(
    image: torch.Tensor,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    target_class: int = None,
) -> np.ndarray:
    """
    Function to calculate Grad-CAM++

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer
        target_class (int | None): Target class

    Returns:
        np.ndarray: Grad-CAM++ heatmap
    """

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

    image = image.unsqueeze(0)  # (1, 3, x, y)

    model.eval()
    output = model(image)

    model.zero_grad()

    if target_class is None:
        output[0, output.argmax()].backward()
    else:
        output[0, target_class].backward()

    model.train()

    # Remove hooks
    h1.remove()
    h2.remove()

    activations = activations.cpu().detach().numpy()
    gradients = gradients.cpu().detach().numpy()

    grads_power_2 = gradients**2
    grads_power_3 = grads_power_2 * gradients
    sum_activations = np.sum(activations, axis=(2, 3))

    alpha = grads_power_2 / (
        2 * grads_power_2 + sum_activations[:, :, None, None] * grads_power_3 + 1e-6
    )
    # alpha coefficient representing the significance of the gradient

    weights = np.sum(alpha * np.maximum(gradients, 0), axis=(2, 3))
    # Calculate the weights as the sum of alpha and the gradients in the (2, 3) axis.
    # These weights are used later to weight the activations

    gradcamplusplus = np.zeros(activations.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        gradcamplusplus += w * activations[0, i, :, :]

    gradcamplusplus = np.maximum(gradcamplusplus, 0)
    gradcamplusplus = cv2.normalize(gradcamplusplus, None, 0, 1, cv2.NORM_MINMAX)

    return gradcamplusplus


def get_xgradcam(
    image: torch.Tensor,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    target_class: int = None,
) -> np.ndarray:
    """
    Function to calculate XGrad-CAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer
        target_class (int | None): Target

    Returns:
        np.ndarray: XGrad-CAM heatmap
    """

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

    image = image.unsqueeze(0)  # Add batch dimension shape (1, 3, x, y)

    model.eval()
    output = model(image)

    model.zero_grad()

    if target_class is None:
        output[0, output.argmax()].backward()
    else:
        output[0, target_class].backward()

    model.train()

    h1.remove()
    h2.remove()

    activations = activations.cpu().detach().numpy()
    gradients = gradients.cpu().detach().numpy()

    weights = np.sum(gradients * activations, axis=(2, 3)) / (
        np.sum(activations, axis=(2, 3)) + 1e-6
    )
    # Dividing the total influence of each channel by the total activation of that channel.
    # This normalizes the weights to account for both the strength of the activation and its impact on the model output.

    xgradcam = np.zeros(activations.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights[0]):
        xgradcam += w * activations[0, i, :, :]

    xgradcam = np.maximum(xgradcam, 0)
    xgradcam = cv2.normalize(xgradcam, None, 0, 1, cv2.NORM_MINMAX)

    return xgradcam


def get_scorecam(
    image: torch.Tensor,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    target_class: int = None,
) -> np.ndarray:
    """
    Function to calculate Score-CAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer
        target_class (int | None): Target class

    Returns:
        np.ndarray: Score-CAM heatmap
    """

    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    h1 = target_layer.register_forward_hook(forward_hook)

    image = image.unsqueeze(0)  # Add batch dimension (1, 3, x, y)

    model.eval()
    _ = model(image)

    h1.remove()

    activations = activations.cpu().detach().numpy()
    upsample = torch.nn.UpsamplingBilinear2d(size=image.shape[-2:])
    activation_tensor = torch.from_numpy(activations)
    activation_tensor = activation_tensor.to(image.device)

    upsampled = upsample(activation_tensor)

    maxs = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=-1)[0]
    mins = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=-1)[0]

    maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
    upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

    input_tensors = image[:, None, :, :] * upsampled[:, :, None, :, :]

    scores = []

    for i in range(input_tensors.size(1)):
        masked_image = input_tensors[:, i, :, :, :]
        with torch.no_grad():
            output = model(masked_image)
        if target_class is not None:
            score = output[0, target_class].item()
        else:
            score = output[0, output.argmax()].item()
        scores.append(score)

    scores = torch.Tensor(scores)
    scores = scores.view(activations.shape[1])
    weights = torch.nn.Softmax(dim=-1)(scores).numpy()

    scorecam = np.zeros(activations.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights):
        scorecam += w * activations[0, i, :, :]

    scorecam = np.maximum(scorecam, 0)
    scorecam = cv2.normalize(scorecam, None, 0, 1, cv2.NORM_MINMAX)

    return scorecam


def get_cam(
    image: torch.Tensor,
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    target_class: int = None,
) -> np.ndarray:
    """
    Function to calculate CAM

    Args:
        image (torch.Tensor): Image tensor with shape (3, x, y)
        model (torch.nn.Module): Model
        target_layer (torch.nn.Module): Target layer
        target_class (int | None): Target class

    Returns:
        np.ndarray: CAM
    """

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

    image = image.unsqueeze(0)  # Add batch dimension shape (1, 3, x, y)

    model.eval()
    output = model(image)

    model.zero_grad()

    if target_class is None:
        output[0, output.argmax()].backward()
    else:
        output[0, target_class].backward()

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
    cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)

    return cam
