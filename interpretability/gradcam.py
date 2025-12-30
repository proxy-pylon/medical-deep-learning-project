import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def generate_cam(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def __del__(self):
        self.remove_hooks()


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    model_name = model_name.lower()
    
    if model_name == 'resnet50':
        return model.backbone.layer4[-1]
    elif model_name == 'senet':
        return model.backbone.layer4[-1]
    elif model_name == 'efficientnet':
        return model.backbone.features[-1]
    elif model_name == 'convnext':
        return model.backbone.features[-1][-1]
    elif model_name == 'vgg16':
        return model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5, 
                   colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlayed = heatmap * alpha + img * (1 - alpha)
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    
    return overlayed


def denormalize_image(img: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406],
                     std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    img = img.cpu().numpy().transpose(1, 2, 0)
    
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img