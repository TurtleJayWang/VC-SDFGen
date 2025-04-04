import torch
import torch.nn as nn

class GridEmbedding(nn.Module):
    def __init__(self, n, grid_size, latent_dim):
        super(GridEmbedding, self).__init__()
        self.grid_size = grid_size

        self.embeddings = nn.Parameter(torch.randn(n, grid_size + 1, grid_size + 1, grid_size + 1, latent_dim))
        self.register_parameter('embeddings', self.embeddings)

    def forward(self, indices):
        n_models = indices.shape[0]
        embeddings = self.embeddings[indices].view(
            n_models, 
            self.grid_size + 1, self.grid_size + 1, self.grid_size + 1, 
            -1
        )
        return embeddings