from torchvision import models, transforms
from torchsummary import summary
import torch
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt

def cam(model, target_layer, image):
    '''
    get model, targer_layer, image
    get activation from target_layer
    get weights from target_layer
    return weights, activation
    '''
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)


    hook = target_layer.register_forward_hook(hook_fn)
    model.eval()
    model(image)
    hook.remove()
    activations = activations[0].detach().numpy()

    weights = activations.mean(axis=(2, 3))
    return weights, activations

def grad_cam(model, target_layer, image):
    activations = []
    gradients = []

    def hook_fn(module, input, output):
        activations.append(output)
    
    def hook_fn2(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    hook = target_layer.register_forward_hook(hook_fn)
    hook2 = target_layer.register_backward_hook(hook_fn2)

    model.eval()
    model.zero_grad()
    output = model(image)
    output[0, output.argmax()].backward()
    hook.remove()
    hook2.remove()

    activations = activations[0].detach().numpy()
    gradients = gradients[0].detach().numpy()

    weights = gradients.mean(axis=(2, 3))
    return weights, activations

def hires_cam(model, target_layer, image):
    activations = []
    gradients = []

    def hook_fn(module, input, output):
        activations.append(output)

    def hook_fn2(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    hook = target_layer.register_forward_hook(hook_fn)
    hook2 = target_layer.register_backward_hook(hook_fn2)

    model.eval()
    model.zero_grad()
    output = model(image)
    output[0, output.argmax()].backward()
    hook.remove()
    hook2.remove()

    activations = activations[0].detach().numpy()
    gradients = gradients[0].detach().numpy()

    weights = gradients.mean(axis=(2, 3))
    return weights, activations

def ablation_cam():
    pass


def make_heatmap(activation, weights):
    '''
    create zeroed heatmap of shape activation.shape[2:]
    do for loop over all filters
    add weighted activation to heatmap
    return heatmap
    '''
    heatmap = np.zeros(activation.shape[2:])
    for i, w in enumerate(weights[0]):
        heatmap += w * activation[0, i, :, :]
    return heatmap



def template(image, model, cam, target_layer, transform):
    '''
    get image
    transform image
    call cam on image
    '''

    weights, activation = cam(model, target_layer, image)
    heatmap = make_heatmap(activation, weights)

    return heatmap

def get_image(path, transform):
    '''
    get image from path and transform it tensor
    '''
    img = PIL.Image.open(path)
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)
    img_it = transform(img)
    return img_t


def overlay_heatmap(image, heatmap):
    '''
    get image and heatmap
    normalize heatmap
    overlay heatmap on image
    return image
    '''
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image = np.uint8(255 * image)
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return overlay


def make_plot(image, heatmap):
    '''
    get image and heatmap
    plot image and heatmap
    '''


    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title('Image')

    axs[2].imshow(heatmap)
    axs[2].axis('off')
    axs[2].set_title('Heatmap')

    axs[1].imshow(overlay_heatmap(image, heatmap))
    axs[1].axis('off')
    axs[1].set_title('Overlay')

    plt.show()

if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    img = get_image('prezentacja2/images/nosacz.jpg', transform)
    
    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[-1].conv3
    
    for enum in [grad_cam]:
        heatmap = template(img, model, enum, target_layer, transform)
        make_plot(img.squeeze().permute(1, 2, 0).numpy(), heatmap)
    
