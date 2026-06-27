import torch
import torch.nn as nn
from torchvision import models

model= models.resnet18(weights='IMAGENET1K_V1')
model.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3, bias=False)
model.fc=nn.Linear(512,1)
fake_input=torch.randn(1,1,224,224)
output=model(fake_input)
sigmoid_output=torch.sigmoid(output)
print("Output shape:", sigmoid_output.shape)
print("Output value:", sigmoid_output.item())
