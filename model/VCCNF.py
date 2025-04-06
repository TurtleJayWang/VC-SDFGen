import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint

class VoxelEncoder(nn.Module):
    def __init__(self, voxel_size, voxel_embedding_dim):
        super(VoxelEncoder, self).__init__()
        
        self.voxel_size = voxel_size
        self.voxel_feature_cnn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            
            nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            
            nn.Conv3d(in_channels=32, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            
            nn.Flatten()
        )
        
        self.fc = nn.Linear((voxel_size // 64) ** 3 * 256, voxel_embedding_dim)
        
    def forward(self, voxel_grid):
        x = voxel_grid.unsqueeze(1)  # Add channel dimension
        x = self.voxel_feature_cnn(x)
        x = F.relu(self.fc(x))
        return x

class ODENetwork(nn.Module):
    def __init__(self, voxel_embedding_dim, latent_dim, hidden_dim=512, num_hidden_layers=4, skip_layers=[1, 3]):
        super(ODENetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Linear(latent_dim + voxel_embedding_dim + 1, hidden_dim),
            nn.ReLU()
        ))
        
        for i in range(num_hidden_layers):
            if i in skip_layers:
                self.layers.append(nn.Sequential(
                    nn.Linear(hidden_dim + voxel_embedding_dim + 1, hidden_dim),
                    nn.ReLU()
                ))
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ))
        
        self.layers.append(nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()  # Assuming the latent space is normalized to [-1, 1]
        ))
        
    def forward(self, voxel_embedding, t, latent):
        for i, layer in enumerate(self.layers):
            if i - 1 in [1, 3]:
                latent = torch.cat([latent, voxel_embedding, t.unsqueeze(1)], dim=1)
            else:
                latent = layer(latent)
        return latent

class VCCNFWrapper(nn.Module):
    def __init__(self, voxel_embedding_dim=128, latent_dim=512, voxel_size=64, hidden_dim=512, num_hidden_layers=4, skip_layers=[1, 3], reverse=False):
        super(VCCNFWrapper, self).__init__()
        
        self.latent_dim = latent_dim
        self.voxel_embedding_dim = voxel_embedding_dim
        self.voxel_size = voxel_size
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        self.encoder = VoxelEncoder(voxel_size, voxel_embedding_dim)
        self.ode_network = ODENetwork(voxel_embedding_dim, latent_dim, hidden_dim, num_hidden_layers, skip_layers)
        self.integrated_time = torch.tensor([0, 1], dtype=torch.float32)  # Time range for integration
        if reverse:
            self.integrated_time = self.integrated_time.flip(0)

    def forward(self, voxel_grid, latent):
        voxel_embedding = self.encoder(voxel_grid)
        def ode_func(t, latent):
            return self.ode_network(voxel_embedding, t, latent)
        return odeint(ode_func, latent, self.integrated_time, method="rk4")[-1]
