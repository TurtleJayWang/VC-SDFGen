import torch
import torch.nn as nn
import torch.nn.functional as F

from model.VoxelSDF import VoxelSDF

from data.dataset import ShapeNetSDF

import numpy as np
import os

class VoxelSDFTraining:
    def __init__(self, voxelsdf_model : VoxelSDF, dataset_path, result_dir, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.dataset = ShapeNetSDF(dataset_path)

        self.model = voxelsdf_model
        self.latent_grid_size = self.model.voxel_grid_size
        self.latent_dim = self.model.latent_dim

        self.embedding_length = len(self.dataset) * (self.latent_grid_size + 1) ** 3
        self.embeddings = nn.Embedding(self.embedding_length, self.latent_dim)

        self.optimizer = torch.optim.Adam([
            { "params": self.model.parameters(), "lr": 1e-4},
            { "params": self.embeddings.parameters(), "lr": 1e-4}
        ])

        self.criterion = nn.L1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_batch_size(self, batch_size):   
        self.batch_size = batch_size

    def __iter__(self):
        self.model.to(self.device)
        
        losses = []
        if os.path.exists(os.path.join(self.result_dir, "losses.npy")):
            losses = np.load(os.path.join(self.result_dir, "losses.npy")).tolist()
        
        start_epoch = self.latest_epoch()
        if start_epoch > 0:
            self.load_model(start_epoch)
        
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for e in range(start_epoch, self.epochs):
            epoch_loss = 0
            for i, (points, sdfs, index) in enumerate(dataloader):
                points = points.to(self.device)
                sdfs = sdfs.to(self.device)

                real_batch_size = index.shape[0]

                embedding_indices = index * (self.latent_grid_size + 1) ** 3
                embedding_indices = embedding_indices.view(real_batch_size, -1)
                embedding_indices = embedding_indices.repeat(1, (self.latent_grid_size + 1) ** 3)
                embedding_indices += torch.arange(0, (self.latent_grid_size + 1) ** 3).unsqueeze(0).repeat(real_batch_size, 1)
                embedding_indices = embedding_indices.view(-1)
                latent_codes = self.embeddings(embedding_indices)
                latent_codes = latent_codes.to(self.device)

                voxel_sdf = self.model(latent_codes, points)

                voxel_sdf = voxel_sdf.view(real_batch_size, -1)
                sdfs = sdfs.view(real_batch_size, -1)
                loss = self.criterion(voxel_sdf, sdfs)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if i % 10 == 0:
                    self.save_model(e)
                    np.save(os.path.join(self.result_dir, "losses.npy"), np.array(losses))
            
            losses.append(epoch_loss)
            yield e, epoch_loss

    def __len__(self):
        return self.epochs

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f"voxel_sdf_epoch_{epoch}.param")
        torch.save(self.embeddings.state_dict(), f"embeddings_epoch_{epoch}.param")

    def load_model(self, epoch):
        self.model.load_state_dict(torch.load(f"voxel_sdf_epoch_{epoch}.param"))
        self.embeddings.load_state_dict(torch.load(f"embeddings_epoch_{epoch}.param"))

    def latest_epoch(self):
        if os.path.exists(os.path.join(self.result_dir, "losses.npy")):
            losses = np.load(os.path.join(self.result_dir, "losses.npy")).tolist()
            latest_epoch =  len(losses) // 10 * 10
            model_embedding_exists = os.path.exists(f"voxel_sdf_epoch_{latest_epoch}.param") and os.path.exists(f"embeddings_epoch_{latest_epoch}.param")
            while not model_embedding_exists:
                latest_epoch -= 10
                if latest_epoch < 0:
                    return 0
                model_embedding_exists = os.path.exists(f"voxel_sdf_epoch_{latest_epoch}.param") and os.path.exists(f"embeddings_epoch_{latest_epoch}.param")
            return latest_epoch
        else:
            return 0
