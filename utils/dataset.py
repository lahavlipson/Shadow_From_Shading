import torch
from utils.scene import Scene
from torch.utils.data import Dataset
import cv2

class ShapeDataset(Dataset):

    def __init__(self, args):
        self.length = args.ep_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sc = Scene()
        sc.add_object()
        shadows, noshadows = sc.render()
        return torch.Tensor(noshadows).permute(2, 0, 1), torch.Tensor(shadows).permute(2, 0, 1)

    def print_numpy(arr, filename):
        cv2.imwrite(filename, arr)

    def print_tensor(tens, filename):
        if tens.shape[0] == 1:
            tens = tens.squeeze(0)
        arr = tens.detach().cpu().permute(1, 2, 0).numpy()
        ShapeDataset.print_numpy(arr,filename)





if __name__ == '__main__':
    ds = ShapeDataset()
