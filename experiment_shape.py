import torch
import torch.utils.data
import os
import numpy as np
from utils.helpers import define_parser, mean, diffs
from shape_net import ShapeNet
from shadow_net import ShadowNet
from utils.dataset import ShapeDataset
from matplotlib import pyplot as plt
from utils.scene import Scene
from torch import nn

class ExperimentShape:


    def __init__(self, args, filepath):
        self.dataset = ShapeDataset(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.workers)
        self.network = ShapeNet()
        self.shadownet = ShadowNet()
        self.shadownet.load_state_dict(torch.load(filepath, map_location='cpu'))

        self.training_losses = []
        self.EPOCHS = args.niter
        self.cuda = args.cuda
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
        for i, (shadowless_views, (true_type, true_loc)) in enumerate(self.dataloader):
            if self.cuda:
                shadowless_views = shadowless_views.cuda()
                true_type = true_type.cuda()
                true_loc = true_loc.cuda()
            self.optimizer.zero_grad()

            encoding = self.shadownet.encode(shadowless_views)
            est_type, est_loc = self.network(encoding)
            training_loss = nn.L2Loss()(est_loc, true_loc) + nn.CrossEntropyLoss()(est_type, true_type)
            running_loss.append(training_loss.item())
            print("Training loss:",str.format('{0:.5f}',mean(running_loss)),"|",str(((i+1)*100)//len(self.dataloader))+"%")
            training_loss.backward()
            self.optimizer.step()

        self.training_losses.append(mean(running_loss))

        if len(self.training_losses) >= 10:
            recent_growth = mean(diffs(self.training_losses[-10:]))
            print("Recent Growth:", recent_growth)

            if False and self.epochs_since_increase >= 10 and 0 < recent_growth < 1:
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
            shadowless_view, _ = self.dataset[0]

            encoding = self.shadownet.encode(shadowless_view.unsqueeze(0))
            est_type, est_loc = self.network(encoding)
            sc = Scene((20,8), True, gridlines_width=20, gridlines_spacing=30)
            sc.add_object(est_type.item())
            sc.shapes[0].center = est_loc.item()
            sc.refocus_camera()
            _, estimated_scene = sc.render()

            ShapeDataset.print_tensor(
                torch.cat([shadowless_view, estimated_scene], 2),
                os.path.join(epoch_folder, "input_estmated_" + str(num) + ".png"))


if __name__ == '__main__':
    filepath = ""
    args = define_parser().parse_args()
    exp = ExperimentShape(args, filepath)
    exp.run()
    print("done")
