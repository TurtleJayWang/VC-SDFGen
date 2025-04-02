import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelSDF(nn.Module):
    def __init__(self, latent_dim=32, voxel_grid_size=8, num_layers=1, hidden_dim=128):
        super(VoxelSDF, self).__init__()
        
        self.sdf_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers)],
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        self.voxel_grid_size = voxel_grid_size
        self.latent_dim = latent_dim

    def forward(self, embeddings, points):
        n_points = points.shape[0]
        points = points.view(1, n_points, 1, 1, 3)

        embeddings = self.embeddings(torch.arange((self.voxel_grid_size + 1) ** 3).long())
        embeddings = embeddings.view(
            1, self.latent_dim,
            self.voxel_grid_size + 1,
            self.voxel_grid_size + 1,
            self.voxel_grid_size + 1
        )

        latent = F.grid_sample(
            embeddings,
            points,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).view(n_points, self.latent_dim)
        
        sdf = self.sdf_mlp(latent)
        return sdf
    
if __name__ == "__main__":
    model = VoxelSDF()
    points = torch.randn(10, 3) 
    sdf_values = model(points)
    print(sdf_values.shape)
