import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelSDF(nn.Module):
    def __init__(self, latent_dim=32, voxel_grid_size=8, num_layers=4, hidden_dim=128):
        super(VoxelSDF, self).__init__()
        
        self.sdf_mlp = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden_dim),
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
        n_models, n_points_per_model = points.shape[0:2]

        points = points.view(n_models, n_points_per_model, 1, 1, 3)

        embeddings = embeddings.view(
            n_models, self.latent_dim,
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
        ).view(n_models, n_points_per_model, self.latent_dim)
        
        points = points.view(n_models, n_points_per_model, 3)
        sdf = self.sdf_mlp(torch.cat([latent, points], dim=-1))
        return sdf