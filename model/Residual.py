import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return self.activation(x + residual)  # Skip connection
