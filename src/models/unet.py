import torch
import torch.nn as nn

class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1=nn.Conv2d(1,8,kernel_size=3,padding=1)
        self.pool1=nn.MaxPool2d(2)
        self.enc_conv2=nn.Conv2d(8,16,kernel_size=3,padding=1)
        self.pool2=nn.MaxPool2d(2)
        self.bottleneck_conv=nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.upsample1=nn.Upsample(scale_factor=2)
        self.dec_conv1=nn.Conv2d(32+16,16,kernel_size=3,padding=1)
        self.upsample2=nn.Upsample(scale_factor=2)
        self.dec_conv2=nn.Conv2d(16+8,8,kernel_size=3,padding=1)
        self.final_conv=nn.Conv2d(8,1,kernel_size=1)

    def forward(self,x):
        e1=torch.relu(self.enc_conv1(x))
        p1=self.pool1(e1)
        e2=torch.relu(self.enc_conv2(p1))
        p2=self.pool2(e2)
        b=torch.relu(self.bottleneck_conv(p2))
        u1=self.upsample1(b)
        d1=torch.relu(self.dec_conv1(torch.cat([u1,e2],dim=1)))
        u2=self.upsample2(d1)
        d2=torch.relu(self.dec_conv2(torch.cat([u2,e1],dim=1)))
        out=self.final_conv(d2)
        out=torch.sigmoid(out)
        return out
    
if __name__=='__main__':
    model=MiniUNet()
    fake_input=torch.randn(1,1,64,64)
    output=model(fake_input)
    print("Output shape:", output.shape)