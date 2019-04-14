import torch
import torch.utils.data
import os
import numpy as np
from utils.helpers import define_parser, mean, diffs
from shadow_vae import ShadowVAE
from utils.dataset import ShapeDataset
from matplotlib import pyplot as plt
from utils.loss_function import shadow_loss, binary_shadow_to_image, binary_shadow, kl_divergence
from utils.vae_loss import loss_function
import torch.nn.functional as F

class Experiment:


    def __init__(self, args):
        self.dataset = ShapeDataset(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.workers)
        self.network = ShadowVAE()
        self.training_losses = []
        self.EPOCHS = args.niter
        self.cuda = args.cuda
        self.pixelwise_loss = shadow_loss
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=args.lr)
        if self.cuda:
            self.network = self.network.cuda()
            #self.pixelwise_loss = self.pixelwise_loss.cuda()

        # create result_dir
        self.results_dir = args.res_dir
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        # create model_dir
        self.model_dir = os.path.join(self.results_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.start = 1  # starting epoch
        if args.net_from_epoch:
            self.load_model(args.net_from_epoch)
            self.start = args.net_from_epoch
            self.training_losses = list(np.loadtxt(os.path.join(self.results_dir, 'total_training_loss.txt')))

        # number of epochs since curriculum update
        self.epochs_since_increase = 10

    def run(self):
        self.evaluate(0, 15)
        for epoch in range(self.start, self.start + self.EPOCHS + 1):
            self.train(epoch)
            self.save_model(epoch)
            self.evaluate(epoch, 15)
            np.savetxt(os.path.join(self.results_dir, 'total_training_loss.txt'), np.array(self.training_losses))
            self.save_graph()

    def save_graph(self):
        plt.clf()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Training Loss")
        plt.grid()
        plt.plot(list(range(1, 1 + len(self.training_losses))), self.training_losses, label='Training Loss')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, 'loss_graph.png'))

    def save_model(self, epoch):
        new_file = '{}/netG_epoch_{}.pth'.format(self.model_dir, epoch)
        torch.save(self.network.state_dict(), new_file)
        old_file = '{}/netG_epoch_{}.pth'.format(self.model_dir, epoch - 2)
        if os.path.exists(old_file):
            os.remove(old_file)

    def load_model(self, epoch):
        self.network.load_state_dict(
            torch.load('{}/netG_epoch_{}.pth'.format(self.model_dir, epoch), \
                       map_location='cpu'))

    def train(self, epoch):
        print("Training Epoch",epoch)

        self.network.train()
        running_loss = []
        for i, (shadowless_views, shadowed_views) in enumerate(self.dataloader):
            if self.cuda:
                shadowless_views = shadowless_views.cuda()
                shadowed_views = shadowed_views.cuda()
            self.optimizer.zero_grad()

            estimated_shadows, mu, logvar = self.network(shadowless_views)
            assert estimated_shadows.shape[1] == 2, estimated_shadows.shape
            threshold = 0.1
            true_binary = binary_shadow(shadowless_views, shadowed_views, threshold)
            pixel_loss = self.pixelwise_loss(true_binary, estimated_shadows)
            true_binary = true_binary.float()
            kld = kl_divergence(estimated_shadows, true_binary)
            print(kld)
            training_loss = pixel_loss + kld
            running_loss.append(training_loss.item())
            print("Training loss:",str.format('{0:.5f}',mean(running_loss)),"|",str(((i+1)*100)//len(self.dataloader))+"%")
            training_loss.backward()
            self.optimizer.step()

        self.training_losses.append(mean(running_loss))

        if len(self.training_losses) >= 10:
            recent_growth = mean(diffs(self.training_losses[-10:]))
            print("Recent Growth:", recent_growth)

            if self.epochs_since_increase >= 10 and 0 < recent_growth < 1:
                self.epochs_since_increase = 0
                if self.dataset.focus:
                    print("STOPPING FOCUS")
                    self.dataset.focus = False
                else:
                    print("INCREASING NUM SHAPES TO", 1 + self.dataset.number_of_shapes)
                    self.dataset.number_of_shapes += 1
            else:
                self.epochs_since_increase += 1




    def evaluate(self, epoch, num_samples):
        print("Evaluation Epoch", str(epoch) + ". Writing", num_samples, "example outputs to", self.results_dir)
        epoch_folder = os.path.join(self.results_dir,"epoch_"+str(epoch))
        if not os.path.isdir(epoch_folder):
            os.mkdir(epoch_folder)

        for num in range(num_samples):
            shadowless_view, shadowed_view = self.dataset[0]
            if self.cuda:
                shadowless_view = shadowless_view.cuda()
                shadowed_view = shadowed_view.cuda()

            estimated_shadow, mu, logvar = self.network(shadowless_view.unsqueeze(0))
            estimated_shadowed_view = binary_shadow_to_image(shadowless_view.unsqueeze(0), estimated_shadow).squeeze(0)

            ShapeDataset.print_tensor(
                torch.cat([shadowless_view, estimated_shadowed_view, shadowed_view], 2).clamp(0.0, 255.0),
                os.path.join(epoch_folder, "input_output_truth_" + str(num) + ".png"))

            #torch.save(estimated_shadowed_view, os.path.join(epoch_folder, "estimate_" + str(num) + ".pt"))


if __name__ == '__main__':
    args = define_parser().parse_args()
    exp = Experiment(args)
    exp.run()
    print("done")
