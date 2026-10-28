import torch 
from torch.utils.data import Dataset
from typing import List
import os

from torchvision.transforms import v2
class FrameTriple(Dataset):
    def __init__(self,path_to_data: List[str]):
        tensors = []
        paths = [os.path.join(path_to_data, f) for f in os.listdir(path_to_data)]
        for i in paths:
            load = torch.load(i, mmap = True)
            triple = torch.stack([load[:-2], load[1:-1], load[2:]], dim = 1) # (B-2, 3, 3, 224, 224)
            tensors.append(triple)

        self.data = torch.cat(tensors, dim = 0)
        self.len = self.data.shape[0]
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        tens = self.data[idx] # (3,3,224, 224)
        total_transform =v2.Compose([v2.ToDtype(torch.float32)])
        return (total_transform(tens) / 255.0)

