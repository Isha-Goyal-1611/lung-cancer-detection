import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key= nn.Linear(embed_dim, embed_dim)
        self.value= nn.Linear(embed_dim, embed_dim)
        pass

    def forward(self,x):
        
        Q=self.query(x)
        K=self.key(x)
        V=self.value(x)

        scores=Q @ K.transpose(-2,-1)
        scores=scores / (x.shape[-1] **0.5)

        weights=F.softmax(scores, dim=-1)
        output=weights @ V
        return output, weights
if __name__=="__main__":
        embed_dim=64
        x= torch.randn(1,16,embed_dim)
        model = SelfAttention(embed_dim)
        output,weights=model(x)
        print("Input shape:",x.shape)
        print("Output shape:", output.shape)
        print("Attention weights shape:",weights.shape)

        plt.figure(figsize=(6, 5))
        plt.imshow(weights[0].detach().numpy(), cmap='hot')
        plt.colorbar()
        plt.title('Attention weights: which patches attend to which?')
        plt.xlabel('Key patches')
        plt.ylabel('Query patches')
        plt.savefig('outputs/day16_attention.png')
        plt.show()

    


        
