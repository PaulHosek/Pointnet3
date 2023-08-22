import wandb

wandb.login()

import getopt
import os.path as osp
import sys

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
import sampling_algs
import utils
import time
import csv

global inputfile


class SAModule(torch.nn.Module):
    """
    set abstraction module, torch.nn.Module = can contain trainable parameters and be optimized during training
    """

    def __init__(self, ratio, r, nn, bias, k):
        """
         nn = nr of output features
        :param ratio:
        :param r: radius within which we sample points
        :param nn:
        """
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.bias = bias
        self.k = k

    def forward(self, x, pos, batch):
        """
        Computation module. Implements sampling, grouping and pointNet layer.
        1. Partition set of points into (possibly) overlapping local regions (using centroids + radius)
        2. Get local neighbourhoods' features for each centroid
        2. Aggregating local neighbourhood using local pointNet layer (shared weights)
        :param x: input features
        :param pos: point position, shape (nr points, 3/xyz)
        :param batch: list of batch indices for all points in the point clouds
        :return:
        """

        # Sampling Layer
        # sample centroids from the point cloud
        # must take in shape [nr points, 3] -> 1D index vector

        # idx = sampling_algs.original_fps(pos, batch, ratio=self.ratio)
        # idx = sampling_algs.wrap_curve(pos, batch, ratio=self.ratio, k=self.k)
        # sampler_args = [self.k]
        # sampler = sampling_algs.max_curve_sampler
        # print(self.ratio)
        # print(self.ratio.shape)
        # idx = sampling_algs.batch_sampling_coordinator(pos, batch, self.ratio, sampler, sampler_args)
        args2 = [self.bias, sampling_algs.max_curve_sampler, self.k]
        func2 = sampling_algs.bias_anyvsfps_sampler
        idx = sampling_algs.batch_sampling_coordinator(pos, batch, self.ratio, func2, args2)

        # Grouping Layer
        # row, col are 1D arrays. If stacked, the columns of the new array represent pairs of points.
        # These pairs of points could represent edges for points within radius r to their respective centroid.
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)

        # select features x of all centroids
        centroids_features_x = None if x is None else x[idx]

        # PointNet Layer
        # get aggregated features by convolution operation on input features;
        # PointNetConv applied to adjacency matrix and input features

        x = self.conv((x if x is None else x, centroids_features_x), (pos, pos[idx]),
                      edge_index)  # FIXME pos[idx] gives problems here

        # Set positions and batch indices to the subset of centroids for the next layer as input
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self, bias, k):
        super().__init__()
        self.bias = bias
        self.k = k

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]), self.bias, self.k)
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]), self.bias, self.k)
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out  # why is this here?

        return self.mlp(x).log_softmax(dim=-1)


def train(epoch, model, train_loader, optimizer, device, loss=False, ):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
    if loss:
        return loss


def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def parse_inputfile(argv):
    # Default values
    inputfile = ''

    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    print('Input file is', inputfile)
    return inputfile


# def parse_args(argv):
#     # Default values
#     inputfile = ''
#     outputfile = ''
#     n_points = 16
#     n_epochs = 1
#     bias = 0.5
#     k = 3
#
#     try:
#         opts, args = getopt.getopt(argv, "hi:o:n:e:b:k:", ["ifile=", "ofile=","n_points=", "n_epochs=", "bias=", "k="])
#     except getopt.GetoptError:
#         print('test.py -i <inputfile> -o <outputfile> -n <n_epochs> -b <bias> -k <k_value>')
#         sys.exit(2)
#
#     for opt, arg in opts:
#         if opt == '-h':
#             print('test.py -i <inputfile> -o <outputfile> -n <n_epochs> -b <bias> -k <k_value>')
#             sys.exit()
#         elif opt in ("-i", "--ifile"):
#             inputfile = arg
#         elif opt in ("-o", "--ofile"):
#             outputfile = arg
#         elif opt in ("-n", "--n_points"):
#             n_points = int(arg)
#         elif opt in ("-e", "--n_epochs"):
#             n_epochs = int(arg)
#         elif opt in ("-b", "--bias"):
#             bias = float(arg)  # Convert the argument to a boolean
#         elif opt in ("-k", "--k"):
#             k = int(arg)
#
#     print('Input file is', inputfile)
#     print('Output file is', outputfile)
#     print('Pointcloud size:', n_points)
#     print('Number of epochs:', n_epochs)
#     print('Bias:', bias)
#     print('Value of k:', k)
#
#     return inputfile, outputfile, n_points, n_epochs, bias, k


# Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a
# model and return metric
# def train_one_epoch(epoch, lr, bs, bias, l):
#   acc = 0.25 + ((epoch/30) +  (random.random()/10))
#   loss = 0.2 + (1 - ((epoch-1)/10 +  random.random()/5))
#   return acc, loss
#
# def evaluate_one_epoch(test_loader):
#     # get acc
#     return pn2c.test(test_loader)
#   acc = 0.1 + ((epoch/20) +  (random.random()/10))
#   loss = 0.25 + (1 - ((epoch-1)/10 +  random.random()/6))
#   return acc, loss


# def main():
#     run = wandb.init()
#     lr = wandb.config.lr
#     epochs = wandb.config.epochs
#
#     for epoch in np.arange(1, epochs):
#         train_acc, train_loss = train_one_epoch(epoch, lr)
#         val_acc, val_loss = evaluate_one_epoch(epoch)
#
#         wandb.log({
#             'epoch': epoch,
#             'train_acc': train_acc,
#             'train_loss': train_loss,
#             'n_points': n_points,
#             'val_acc': val_acc,
#             'val_loss': val_loss
#         })


def main():
    run = wandb.init()
    lr = wandb.config.lr
    n_points = wandb.config.n_points
    n_epochs = wandb.config.epochs
    bias = wandb.config.bias
    k = wandb.config.k

    print('lr', lr)
    print('Pointcloud size:', n_points)
    print('Number of epochs:', n_epochs)
    print('Bias:', bias)
    print('Value of k:', k)

    # lr  =  .001
    # n_points = 16
    # n_epochs = 3
    # bias = 0.5
    # k = 3

    print(inputfile)

    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(n_points)  # 1024
    train_dataset = ModelNet(inputfile, '10', True, transform, pre_transform)
    test_dataset = ModelNet(inputfile, '10', False, transform, pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)  # 6 workers
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)  # 6 workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = Net(bias=bias, k=k).to(device)
    # wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("net build")

    accuracies = torch.zeros(n_epochs)
    for epoch in range(1, n_epochs + 1):  # 201
        before = time.time()
        # train(epoch, model=model)
        # test_acc = test(test_loader, model=model)
        train_loss = train(epoch, model, train_loader, optimizer, device, loss=True)
        test_acc = test(test_loader, model, device)
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_acc': test_acc,
            "Duration:": time.time() - before,
        })

        print(f'Epoch: {epoch}, Test: {test_acc}')
        print("Duration:", time.time() - before)
        accuracies[epoch - 1] = test_acc


# inputfile, _, _, _, _, _ = pn2c.parse_args(sys.argv[1:])

# inputfile = "/Users/paulhosek/PycharmProjects/GeometricDL/../data/ModelNet10"

if __name__ == "__main__":
    inputfile = parse_inputfile(sys.argv[1:])
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep_2',
        'metric': {'goal': 'maximize', 'name': 'test_acc'},
        'parameters':
            {
                'epochs': {'values': [7]},
                'inputfile': {'values': [inputfile]},
                'n_points': {'values': [32, 128, 512, 1024]},  # TODO shorter for testing, add rest later
                'lr': {'values': [.001]},
                # "bias": {'max': 1, 'min': 0},
                "bias": {'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
                "k": {'max': 20, 'min': 3}
            }
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project='full_sweep'
    )
    print(inputfile)

    wandb.agent(sweep_id, function=main, count=1000)
