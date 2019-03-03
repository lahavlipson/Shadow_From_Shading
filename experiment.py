import torch
import torch.utils.data
import os
from utils.helpers import define_parser, mean
from dataset import ShapeDataset

class Experiment:


    def __init__(self, args):
        self.dataset = ShapeDataset(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.workers)

        self.training_losses = []


    def run(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass




if __name__ == '__main__':
    args = define_parser().parse_args()
    exp = Experiment(args)
    print("done")