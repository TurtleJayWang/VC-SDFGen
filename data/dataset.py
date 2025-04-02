import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ShapeNetSDF(Dataset):
    def __init__(self, path, used_categories="All"):
        category_names = os.listdir(path)
        self.points, self.sdfs = [], []
        for category_name in category_names:
            model_file_names = os.listdir(os.path.join(path, category_name))
            for model_file_name in model_file_names:
                points, sdfs = self.load_npz_file(os.path.join(path, category_name, model_file_name))
                self.points.append(points)
                self.sdfs.append(sdfs)
        self.n_models = len(self.points)

    def __len__(self):
        return self.n_models

    def __getitem__(self, index):
        return self.points[index], self.sdfs[index], index

    def load_npz_file(self, file_path):
        data = np.load(file_path)
        return torch.from_numpy(data["points"]), torch.from_numpy(data["sdfs"])
