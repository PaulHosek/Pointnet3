import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius


class SAModule(torch.nn.Module):
    """
    set abstraction module, torch.nn.Module = can contain trainable paramters and be optimized during training
    """
    def __init__(self, ratio, r, nn):
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
        idx = fps(pos, batch, ratio=self.ratio)
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
        x = self.conv((x, centroids_features_x), (pos, pos[idx]), edge_index)

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
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)


def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == '__main__':
    # path = "/var/scratch/pmms2305/ModelNet10"
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
    #                                 'data/ModelNet10')
    path = "/var/scratch/pmms2305/ModelNet10"

    print(path)
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(256)  # 1024
    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4)  # 6 workers
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=4)  # 6 workers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 90):  # 201
        train(epoch)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch}, Test: {test_acc}')

    torch.save(model.state_dict(), 'model_fps.pt')
