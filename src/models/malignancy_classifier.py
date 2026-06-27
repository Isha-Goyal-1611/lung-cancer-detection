import torch
import torch.nn as nn
class MalignancyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv3d(1,16, kernel_size=3, padding=1)
        self.conv2=nn.Conv3d(16,32, kernel_size=3, padding=1)
        self.conv3=nn.Conv3d(32,64, kernel_size=3, padding=1)
        self.pool=nn.MaxPool3d(2)
        self.fc1=nn.Linear(64*4*4*4,256)
        self.fc2=nn.Linear(256,64)
        self.clinical_fc=nn.Linear(3,16)
        self.final_fc=nn.Linear(80,1)
        self.dropout=nn.Dropout(0.3)

    def forward(self, patch, clinical_features):
        x=torch.relu(self.conv1(patch))
        x=self.pool(x)
        x=torch.relu(self.conv2(x))
        x=self.pool(x)
        x=torch.relu(self.conv3(x))
        x=self.pool(x)
        x=x.view(x.size(0), -1)
        x=torch.relu(self.fc1(x))
        x=self.dropout(x)
        x=torch.relu(self.fc2(x))
        c=torch.relu(self.clinical_fc(clinical_features))
        combined=torch.cat([x,c], dim=1)
        output=torch.sigmoid(self.final_fc(combined))
        return output
    
if __name__=="__main__":
    model=MalignancyClassifier()
    fake_patch=torch.randn(1,1,32,32,32)
    fake_clinical=torch.tensor([[6.5, -450.0, 120.0]])
    output=model(fake_patch,fake_clinical)
    print("Malignancy score:",output.item())
    print("Output shape:",output.shape)
    score=output.item()
    if score<0.3:
        print("Lung-RADS: 1-2(Routine follow-up)")
    elif score<0.6:
        print("Lung-RADS: 3 (3-month follow-up CT)")
    elif score<0.8:
        print("Lung-RADS: 4A (PET scan recommended)")
    else:
        print("Lung-RADS: 4B (Biopsy recommended)")

