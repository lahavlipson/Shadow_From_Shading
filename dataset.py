import torch
from scene import Scene
from torch.utils.data import Dataset

class ShapeDataset(Dataset):

    def __init__(self, args):
        self.length = args.ep_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sc = Scene()
        sc.add_object()
        shadows, noshadows = sc.render()
        return torch.Tensor(shadows), torch.Tensor(noshadows)



if __name__ == '__main__':
    ds = ShapeDataset()
