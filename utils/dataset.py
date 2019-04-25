import torch
from utils.scene import Scene
from torch.utils.data import Dataset
import cv2
import numpy as np

class ShapeDataset(Dataset):

    def __init__(self, args):
        self.length = args.ep_len
        self.focus = True
        self.number_of_shapes = 1
        self.variability = (20,8)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sc = Scene(self.variability, True, gridlines_width=20, gridlines_spacing=30)
        for _ in range(self.number_of_shapes):
            sc.add_object()
        sc.ground_mesh()
        sc.refocus_camera()
        sc.mutate_all_objects()
        sc.ground_mesh()
        if self.focus:
            sc.refocus_camera()

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
