import torch
import torch.utils.data
import os
from shadow_net import ShadowNet
from utils.helpers import define_parser, mean
from utils.dataset import ShapeDataset
import cv2

class Experiment:


    def __init__(self, args):
        self.dataset = ShapeDataset(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.workers)
        self.network = ShadowNet()
        self.training_losses = []
        self.EPOCHS = args.niter


    def run(self):

        for epoch in range(1, self.EPOCHS + 1):
            self.train(epoch)

    def train(self, epoch):
        print("Training Epoch",epoch)
        print(len(self.dataset))
        for i, (shadowless_views, shadowed_views) in enumerate(self.dataloader):
            print("Iteration",i, shadowless_views[0].squeeze(0).shape, shadowless_views.min(), shadowless_views.max())
            ShapeDataset.print_tensor(shadowless_views[0], "shadowless_views.png")
            #estimated_shadowed_views = self.network(shadowless_views)


    def evaluate(self, num_samples):
        pass




if __name__ == '__main__':
    args = define_parser().parse_args()
    exp = Experiment(args)
    exp.run()
    print("done")