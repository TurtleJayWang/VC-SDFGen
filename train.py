import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from model.VoxelSDF import VoxelSDF
from model.DeepSDF import DeepSDF
from model.VCCNF import VCCNFWrapper
from data.dataset import ShapeNetSDF, ShapeNetVoxel64
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
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)

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
            
        losses = losses[:start_epoch]
        self.scheduler.last_epoch = start_epoch
        
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

            self.scheduler.step()

            if e % 100 == 0:
                self.save_model(e)
                np.save(os.path.join(self.result_dir, "losses.npy"), np.array(losses))

            losses.append(epoch_loss)
            yield e, losses

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
        
class DeepSDFTraining:
    def __init__(self, deepsdf_model : VoxelSDF, dataset_path, result_dir, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.dataset = ShapeNetSDF(dataset_path)

        self.model = deepsdf_model
        self.latent_dim = self.model.latent_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")

        self.embeddings = nn.Embedding(len(self.dataset), self.latent_dim)
        torch.nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)  # Initialize embeddings
        self.embeddings = self.embeddings.to(self.device)

        self.optimizer = torch.optim.Adam([
            { "params": self.model.parameters(), "lr": 1e-3 },
            { "params": self.embeddings.parameters(), "lr": 1e-3 }
        ])
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.5)

        self.criterion = nn.L1Loss(reduction="sum")
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
            
        losses = losses[:start_epoch]
        self.scheduler.last_epoch = start_epoch
        
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for e in range(start_epoch, self.epochs + 1):
            epoch_loss = 0
            for i, (points, sdfs, index) in enumerate(dataloader):
                points = points.to(self.device)
                sdfs = sdfs.to(self.device)

                real_batch_size = index.shape[0]

                latent_codes = self.embeddings(index.to(self.device))

                sdf_pred = self.model(latent_codes, points)

                sdf_pred = sdf_pred.view(real_batch_size, -1)
                sdfs = sdfs.view(real_batch_size, -1)
                loss = self.criterion(sdf_pred, sdfs)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            self.scheduler.step()

            if e % 100 == 0:
                self.save_model(e)
                np.save(os.path.join(self.result_dir, "losses.npy"), np.array(losses))

            losses.append(epoch_loss)
            yield e, losses

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
        
class VCCNFTraining:
    def __init__(self, vccnf_model : VCCNFWrapper, dataset_path, result_dir, epochs, batch_size, embedding_epoch, embedding_dir):
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.dataset = ShapeNetVoxel64(dataset_path)
        self.shapenetvoxel64_dataset_splits = random_split(
            self.dataset, 
            [0.8, 0.2], 
            generator=torch.Generator().manual_seed(42)
        )
        self.shapenetvoxel64_loader_training = DataLoader(
            self.shapenetvoxel64_dataset_splits[0],
            batch_size=batch_size,
            shuffle=True
        )

        self.model = vccnf_model
        self.latent_dim = self.model.latent_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")

        self.embeddings = nn.Embedding(len(self.dataset), self.latent_dim)
        self.embeddings.load_state_dict(
            torch.load(
                os.path.join(embedding_dir, f"embeddings_epoch_{embedding_epoch}.param"), 
                weights_only=True
            )
        )
        self.embeddings = self.embeddings.to(self.device)
        
        self.optimizer = torch.optim.Adam(vccnf_model.parameters(), lr=1e-3)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)
        
        self.losses = []
        if os.path.exists(os.path.join(result_dir, "losses_vccnf.npy")):
            self.losses = np.load(os.path.join(result_dir, "losses_vccnf.npy")).tolist()
        self.start_epoch = self.latest_epoch()
        self.losses = self.losses[:self.start_epoch]
        
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
            
        self.loss_fn = nn.MSELoss()
        
    def __iter__(self):
        for e in range(self.start_epoch, self.epochs + 1):
            epoch_loss = 0
            self.model.train()
            for i, (voxel_data, indices) in enumerate(self.shapenetvoxel64_loader_training):
                indices = indices.to(self.device)
                latent_codes = self.embeddings(indices)
                latent_codes = latent_codes.to(self.device)
                
                gaussian_latents = torch.randn_like(latent_codes)
                gaussian_latents = gaussian_latents.to(self.device)
                
                latent_pred = self.model(voxel_data, gaussian_latents)

                loss = self.loss_fn(latent_pred, latent_codes)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            self.scheduler.step()
            
            self.losses.append(epoch_loss)
            if (e + 1) % 100 == 0:
                print(f"Epoch {e + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")
                np.save(os.path.join(self.result_dir, "losses_vccnf.npy"), np.array(self.losses))
                self.save_model(e + 1)
            yield e, self.losses
        
    def save_model(self, epoch):
        model_path = os.path.join(self.result_dir, f"vccnf_epoch_{epoch}.param")
        torch.save(self.model.state_dict(), model_path)
        
    def __len__(self):
        return self.epochs - self.start_epoch
        
    def latest_epoch(self):
        if os.path.exists(os.path.join(self.result_dir, "losses_vccnf.npy")):
            losses = np.load(os.path.join(self.result_dir, "losses_vccnf.npy")).tolist()
            latest_epoch = len(losses) // 100 * 100
            return latest_epoch
        else:
            return 0
            
    def kl_normal_loss_fn(self, output_embedding):
        mean = torch.mean(output_embedding, dim=0)
        std = torch.std(output_embedding, dim=0)
        var = std ** 2
        kl_per_dim = 0.5 * (mean**2 + var - torch.log(var + 1e-8) - 1)
        kl_total = torch.sum(kl_per_dim)
        return kl_total 
            