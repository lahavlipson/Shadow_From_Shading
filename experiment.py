import torch
import torch.utils.data
import os
from shadow_net import ShadowNet
from utils.helpers import define_parser, mean
from utils.dataset import ShapeDataset

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
        self.evaluate(0, 7)
        for epoch in range(1, self.EPOCHS + 1):
            self.train(epoch)
            self.evaluate(epoch, 7)

    def train(self, epoch):
        print("Training Epoch",epoch)
        self.network.train()
        running_loss = []
        for i, (shadowless_views, shadowed_views) in enumerate(self.dataloader):
            if self.cuda:
                shadowless_views = shadowless_views.cuda()
                shadowed_views = shadowed_views.cuda()
            self.optimizer.zero_grad()

            estimated_shadows = self.network(shadowless_views)
            estimated_shadowed_views = (shadowless_views - estimated_shadows).clamp(0.0, 255.0)
            training_loss = self.pixelwise_loss(estimated_shadowed_views, shadowed_views)
            running_loss.append(training_loss.item())
            print("Training loss:",str.format('{0:.5f}',mean(running_loss)),"|",str(((i+1)*100)//len(self.dataloader))+"%")
            training_loss.backward()
            self.optimizer.step()

        self.training_losses.append(mean(running_loss))


    def evaluate(self, epoch, num_samples):
        print("Evaluation Epoch", str(epoch) + ". Writing", num_samples, "example outputs to tmp_scenes/")
        if not os.path.isdir("tmp_scenes"):
            os.mkdir("tmp_scenes")
        epoch_folder = os.path.join("tmp_scenes","epoch_"+str(epoch))
        if not os.path.isdir(epoch_folder):
            os.mkdir(epoch_folder)

        for num in range(num_samples):
            shadowless_view, shadowed_view = self.dataset[0]
            if self.cuda:
                shadowless_view = shadowless_view.cuda()
                shadowed_view = shadowed_view.cuda()

            estimated_shadow = self.network(shadowless_view.unsqueeze(0)).squeeze(0)
            estimated_shadowed_view = (shadowless_view - estimated_shadow).clamp(0.0, 255.0)
            ShapeDataset.print_tensor(shadowless_view, os.path.join(epoch_folder, "network_input" + str(num) + ".png"))
            ShapeDataset.print_tensor(shadowed_view, os.path.join(epoch_folder,"ground_truth_" + str(num) + ".png"))
            ShapeDataset.print_tensor(estimated_shadowed_view, os.path.join(epoch_folder, "network_output" + str(num) + ".png"))




if __name__ == '__main__':
    args = define_parser().parse_args()
    exp = Experiment(args)
    exp.run()
    print("done")
