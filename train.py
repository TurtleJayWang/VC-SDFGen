import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from model.VoxelSDF import VoxelSDF
from data.dataset import ShapeNetSDF
from model.embedding import GridEmbedding

import numpy as np
import os

class StepLearningRateSchedule:
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class VoxelSDFTraining:
    def __init__(self, voxelsdf_model : VoxelSDF, dataset_path, result_dir, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.dataset = ShapeNetSDF(dataset_path)

        self.model = voxelsdf_model
        self.latent_grid_size = self.model.voxel_grid_size
        self.latent_dim = self.model.latent_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")

        self.embeddings = GridEmbedding(len(self.dataset), self.latent_grid_size, self.latent_dim)
        self.embeddings = self.embeddings.to(self.device)

        self.optimizer = torch.optim.Adam([
            { "params": self.model.parameters(), "lr": 1e-4 },
            { "params": self.embeddings.parameters(), "lr": 1e-2 }
        ])
        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.9)

        self.criterion = nn.L1Loss()
        print(self.device)

        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_batch_size(self, batch_size):   
        self.batch_size = batch_size

    def __iter__(self):
        self.model = self.model.to(self.device)
        
        losses = []
        if os.path.exists(os.path.join(self.result_dir, "losses.npy")):
            losses = np.load(os.path.join(self.result_dir, "losses.npy")).tolist()
        
        start_epoch = self.latest_epoch()
        if start_epoch > 0:
            self.load_model(start_epoch)
        
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for e in range(start_epoch, self.epochs + 1):
            epoch_loss = 0
            for i, (points, sdfs, index) in enumerate(dataloader):
                points = points.to(self.device)
                sdfs = sdfs.to(self.device)

                real_batch_size = index.shape[0]

                latent_codes = self.embeddings(index)
                latent_codes = latent_codes.to(self.device)

                voxel_sdf = self.model(latent_codes, points)

                voxel_sdf = voxel_sdf.view(real_batch_size, -1)
                sdfs = sdfs.view(real_batch_size, -1)
                loss = self.criterion(voxel_sdf, sdfs)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if e % 100 == 0:
                self.save_model(e)
                np.save(os.path.join(self.result_dir, "losses.npy"), np.array(losses))

            losses.append(epoch_loss)
            yield e, epoch_loss

    def __len__(self):
        return self.epochs - self.latest_epoch()

    def save_model(self, epoch):
        model_path = os.path.join(self.result_dir, f"voxel_sdf_epoch_{epoch}.param")
        embedding_path = os.path.join(self.result_dir, f"embeddings_epoch_{epoch}.param")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.embeddings.state_dict(), embedding_path)

    def load_model(self, epoch):
        model_path = os.path.join(self.result_dir, f"voxel_sdf_epoch_{epoch}.param")
        embedding_path = os.path.join(self.result_dir, f"embeddings_epoch_{epoch}.param")
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.embeddings.load_state_dict(torch.load(embedding_path, weights_only=True))

    def latest_epoch(self):
        if os.path.exists(os.path.join(self.result_dir, "losses.npy")):
            losses = np.load(os.path.join(self.result_dir, "losses.npy")).tolist()
            latest_epoch =  len(losses) // 100 * 100
            model_path = lambda e : os.path.join(self.result_dir, f"voxel_sdf_epoch_{e}.param")
            embedding_path = lambda e : os.path.join(self.result_dir, f"embeddings_epoch_{e}.param")
            model_embedding_exists = os.path.exists(model_path(latest_epoch)) and os.path.exists(embedding_path(latest_epoch))
            while not model_embedding_exists:
                latest_epoch -= 100
                if latest_epoch < 0:
                    return 0
                model_embedding_exists = os.path.exists(model_path(latest_epoch)) and os.path.exists(embedding_path(latest_epoch))
            return latest_epoch
        else:
            return 0