import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_2d import SimpleCNN

class GradCAM:
    def __init__(self, model, target_layer):
        self.model=model
        self.target_layer=target_layer
        self.gradients=None
        self.activations=None
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations=output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor):
        output=self.model(input_tensor)
        self.model.zero_grad()
        output.backward()
        weights = self.gradients.mean(dim=[2,3],keepdim=True)
        heatmap=(weights*self.activations).sum(dim=1, keepdim=True)
        heatmap=torch.relu(heatmap)
        heatmap=heatmap.squeeze().numpy()
        if heatmap.max()>0:
            heatmap=heatmap/heatmap.max()

        return heatmap
    
if __name__=="__main__":
    model=SimpleCNN()
    model.eval()
    fake_input=torch.randn(1,1,32,32,requires_grad=True)
    grad_cam=GradCAM(model,model.conv2)
    heatmap=grad_cam.generate_heatmap(fake_input)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(fake_input[0,0].detach().numpy(), cmap='gray')
    plt.title('Input CT patch')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Grad-CAM heatmap\n(bright=model focused here)')
    plt.colorbar()
    plt.savefig('outputs/day_18_gradcam.png')
    plt.show()

    print("heatmap shape:", heatmap.shape)
    print("heatmap min/max:", heatmap.min().round(3),heatmap.max().round(3))