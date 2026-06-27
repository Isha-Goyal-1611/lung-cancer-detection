import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8, kernel_size=3)
        self.conv2 = nn.Conv2d(8,16, kernel_size=3)
        self.fc1 = nn.Linear(16*28*28, 64)
        self.fc2 = nn.Linear(64, 1)
        pass

    def forward(self, x):
        x=self.conv1(x)
        x = torch.relu(x)
        x=self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x=self.fc1(x)
        x = torch.relu(x)
        x=self.fc2(x)
        x=torch.sigmoid(x)

        return x
        pass

if __name__=='__main__':
    model=SimpleCNN()

    fake_input=torch.randn(1,1,32,32)
    output=model(fake_input)
    print("Output shape:", output.shape)    
    