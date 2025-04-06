import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSDF(nn.Module):
    def __init__(self, latent_dim=512, num_hidden_layers=8, hidden_dim=1024, skip_layers=[3, 7]):
        super(DeepSDF, self).__init__()
        
        self.sdf_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + 3, hidden_dim),
                nn.ReLU()
            )])
        for i in range(num_hidden_layers):
            if i in skip_layers:
                self.sdf_mlp.append(nn.Sequential(
                    nn.Linear(hidden_dim + latent_dim + 3, hidden_dim),
                    nn.ReLU()
                ))
            else:
                self.sdf_mlp.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ))        
        self.sdf_mlp.append(nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        ))  # Output layer for SDF
        
        self.dropout = nn.Dropout(0.2)
        
        self.latent_dim = latent_dim
    
    def forward(self, embeddings, points):
        n_models, n_points_per_model = points.shape[0:2]

        points = points.view(-1, 3)
        embeddings = embeddings.view(n_models, self.latent_dim)
        embeddings = embeddings.unsqueeze(1).repeat(1, n_points_per_model, 1).view(-1, self.latent_dim)

        sdf_inputs = torch.cat([embeddings, points], dim=-1)
        sdf = sdf_inputs
        for i, layer in enumerate(self.sdf_mlp[0:-1]):
            if i - 1 in [3, 7]:
                sdf = torch.cat([sdf_inputs, sdf], dim=-1)
            sdf = self.dropout(layer(sdf))
        sdf = self.sdf_mlp[-1](sdf)
        return sdf.view(-1, 1)