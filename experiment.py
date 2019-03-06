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
        self.cuda = args.cuda
        self.pixelwise_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.lr)
        if self.cuda:
            self.network = self.network.cuda()
            self.pixelwise_loss = self.pixelwise_loss.cuda()

    def run(self):

        for epoch in range(1, self.EPOCHS + 1):
            self.train(epoch)

    def train(self, epoch):
        print("Training Epoch",epoch)
        self.network.train()
        running_loss = []
        for i, (shadowless_views, shadowed_views) in enumerate(self.dataloader):
            if self.cuda:
                shadowless_views = shadowless_views.cuda()
                shadowed_views = shadowed_views.cuda()
            self.optimizer.zero_grad()

            estimated_shadowed_views = self.network(shadowless_views)
            training_loss = self.pixelwise_loss(estimated_shadowed_views, shadowed_views)
            running_loss.append(training_loss.item())
            print("Training loss:",mean(running_loss))
            training_loss.backward()
            self.optimizer.step()

        self.training_losses.append(mean(running_loss))

            # print("Iteration",i, shadowless_views[0].squeeze(0).shape, shadowless_views.min(), shadowless_views.max())
            # ShapeDataset.print_tensor(shadowless_views[0], "shadowless_views.png")

            # print("Finished iteration")

    # def evaluate(self, num_samples):
    #     for i, (shadowless_views, shadowed_views) in enumerate(self.dataloader):





if __name__ == '__main__':
    args = define_parser().parse_args()
    exp = Experiment(args)
    exp.run()
    print("done")