import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import trimesh
from skimage import measure

from model.VoxelSDF import VoxelSDF

class Visualizer:
    def __init__(self, sdf_decoder : VoxelSDF, latent_vecs : nn.Embedding):
        self.sdf_decoder = sdf_decoder
        self.latent_grid_size = self.sdf_decoder.voxel_grid_size
        self.latent_dim = self.sdf_decoder.latent_dim

        self.latent_vecs = latent_vecs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sdf_decoder.to(self.device)

    def get_sdf_for_marching_cube(self, embedding_index):
        # Initialize to mesh grid points
        x, y, z = np.mgrid[-1 : 1 : 0.02, -1 : 1 : 0.02, -1 : 1 : 0.02]
        x = torch.tensor(x)
        y = torch.tensor(y)
        z = torch.tensor(z)

        # Change the mesh grid points from 100x100x100x3 to 1000000x3
        points = torch.stack((x, y, z), dim=3).view(-1, 3).float()
        points = points.to(self.device)

        # Split the points into 4 seperate splits to prevent running out of memory
        points_splits = points.split(250000)

        latent_code = self.latent_vecs(torch.tensor([embedding_index]))
        latent_code = latent_code.view(1, self.latent_dim, self.latent_grid_size + 1, self.latent_grid_size + 1, self.latent_grid_size + 1)
        latent_code = latent_code.to(self.device)

        sdfs = torch.zeros(0, device="cpu")
        for points_split in points_splits:
            points_split = points_split.view(1, 250000, 3)
            sdfs = torch.cat((sdfs, torch.clamp(self.sdf_decoder(latent_code, points_split).cpu(), -0.1, 0.1)))

        sdfs = sdfs.view((100, 100, 100))
        sdfs = sdfs.numpy()
        return sdfs

    def generate_sdf_objs(self, index):
        self.sdf_decoder.eval()
        with torch.no_grad():
            # Get the sdf values from model
            sdfs = self.get_sdf_for_marching_cube(index)
            
            print(f"Volume range: {np.min(sdfs)} to {np.max(sdfs)}")

            # Marching Cube
            verts, faces, normals, _ = measure.marching_cubes(sdfs, 0)
            
            # Output the result into mesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_normals=normals)
            with open(f"mesh_reconstruct_model_{index}.obj", "w") as f:
                mesh.export(f, "obj")