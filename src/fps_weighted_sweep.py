# import wandb

# wandb.login()

import getopt
import sys

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, global_max_pool, radius
import sampling_algs
import principal_curvature
import time

global inputfile


class SAModule(torch.nn.Module):
    """
    set abstraction module, torch.nn.Module = can contain trainable parameters and be optimized during training
    """

    def __init__(self, ratio, r, nn, curvature_scalar):
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
        self.curvature_scalar = curvature_scalar

    def forward(self, x, pos, batch, curvature_values): # add curvature values arg again
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


        # sampler_args = [curvature_values, self.curvature_scalar]
        # sampler = sampling_algs.fps_weighted
        # idx = sampling_algs.batch_sampling_coordinator(pos, batch, self.ratio, sampler, sampler_args)

        sampler_args = [curvature_values, self.curvature_scalar]
        sampler = sampling_algs.fps_weighted
        idx = sampling_algs.batch_sampling_coordinator(pos, batch, self.ratio, sampler, sampler_args,
                                                       pass_cloud_idx=True)

        # args2 = [.5, sampling_algs.max_curve_sampler, 10]
        # func2 = sampling_algs.bias_anyvsfps_sampler
        # idx = sampling_algs.batch_sampling_coordinator(pos, batch, self.ratio, func2, args2)



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
        curvature_values = curvature_values[idx]
        return x, pos, batch, curvature_values


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
    def __init__(self, curvature_scalar):
        super().__init__()
        self.curvature_scalar = curvature_scalar

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]), self.curvature_scalar)
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]), self.curvature_scalar)
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, norm=None)

    def forward(self, data):
        # build curvatures
        curvature_values = torch.zeros(len(data.pos))
        points_per_cloud = int(len(data.pos)/len(data.y))
        for cloud_idx in range(0, len(data.y)): # 0..batchsize (e.g., 32)
            curvature_values[cloud_idx*points_per_cloud: (cloud_idx+1)*points_per_cloud] =\
                principal_curvature.curvatures_knn(data[cloud_idx].pos, k=10)

        # sa0_out = (data.x, data.pos, data.batch, curvature_values)
        sa0_out = (data.x, data.pos, data.batch, curvature_values)
        sa1_out = self.sa1_module(*sa0_out)

        # x, pos, batch, _ = self.sa2_module(*sa1_out) # remove the curvature values for the global layer
        x, pos, batch, curvature_values = self.sa2_module(*sa1_out) # remove the curvature values for the global layer
        sa3_out = self.sa3_module(*[x, pos, batch])
        x, pos, batch = sa3_out

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




def main():
    # run = wandb.init()
    # lr = wandb.config.lr
    # n_points = wandb.config.n_points
    # n_epochs = wandb.config.epochs
    # curvature_scalar = wandb.config.curvature_scalar
    # k = wandb.config.k
    #
    # print('lr', lr)
    # print('Pointcloud size:', n_points)
    # print('Number of epochs:', n_epochs)
    # print('topn:', curvature_scalar)
    # print('Value of k:', k)

    lr  =  .001
    n_points = 32
    n_epochs = 3
    curvature_scalar = 10
    k = 3
    ratio = 0.5

    print(inputfile)

    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(n_points)  # 1024
    train_dataset = ModelNet(inputfile, '10', True, transform, pre_transform)
    test_dataset = ModelNet(inputfile, '10', False, transform, pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)  # 6 workers
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)  # 6 workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Net(curvature_scalar=curvature_scalar).to(device)
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
        # wandb.log({
        #     'epoch': epoch,
        #     'train_loss': train_loss,
        #     'test_acc': test_acc,
        #     "Duration:": time.time() - before,
        # })

        print(f'Epoch: {epoch}, Test: {test_acc}')
        print("Duration:", time.time() - before)
        accuracies[epoch - 1] = test_acc


# inputfile, _, _, _, _, _ = pn2c.parse_args(sys.argv[1:])

# inputfile = "/Users/paulhosek/PycharmProjects/GeometricDL/../data/ModelNet10"

if __name__ == "__main__":

    # inputfile = parse_inputfile(sys.argv[1:])
    # sweep_configuration = {
    #     'method': 'random',
    #     'name': 'sweep_2',
    #     'metric': {'goal': 'maximize', 'name': 'test_acc'},
    #     'parameters':
    #         {
    #             'epochs': {'values': [7]},
    #             'inputfile': {'values': [inputfile]},
    #             'n_points': {'values': [1024]},
    #             'lr': {'values': [.001]},
    #             "curvature_scalar": {'values': [1, 8, 16, 32, 64,128]},
    #             "k": {'max': 20, 'min': 3}
    #         }
    # }
    # sweep_id = wandb.sweep(
    #     sweep=sweep_configuration,
    #     project='full_sweep'
    # )
    # print(inputfile)
    #
    # wandb.agent(sweep_id, function=main, count=1000)

    inputfile = "/Users/paulhosek/PycharmProjects/GeometricDL/../data/ModelNet10"
    main()
