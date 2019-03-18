import torch
from utils.scene import Scene
from torch.utils.data import Dataset
import cv2
import numpy as np

class ShapeDataset(Dataset):

    def __init__(self, args):
        self.length = args.ep_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sc = Scene(True, gridlines_width=20, gridlines_spacing=30)
        sc.add_object()
        shadows, noshadows = sc.render()
        shad_tens = torch.Tensor(shadows).permute(2, 0, 1)
        noshad_tens = torch.Tensor(noshadows).permute(2, 0, 1)
        return noshad_tens, shad_tens

    def print_numpy(arr, filename):
        cv2.imwrite(filename, arr)

    def print_tensor(tens, filename):
        arr = tens.detach().cpu().permute(1, 2, 0).numpy()
        ShapeDataset.print_numpy(arr,filename)





if __name__ == '__main__':
    ds = ShapeDataset()
