"""
Various sampling algorithms to sample points in point clouds for use in the PointNet++ sampling layer.
"""

import torch
import torch_cluster

from torch_geometric.typing import OptTensor
import principal_curvature
import scipy.stats as stats
import math
import numpy as np


def max_curve_sampler(cloud,desired_num_points, k):
    """
    Sample points with the highest curvature based on eigenvalues derived from their k-neighbourhood.

    :param cloud: Tensor of shape (N, C) representing the point cloud data.
    :param desired_num_points: Desired number of points to be sampled.
    :param k: Number of nearest neighbors to consider for curvature computation.
    :return: Tensor of shape (M,) containing the indices of the sampled points.
    """
    if cloud.size(0) < k:
        k = cloud.size(0)

    knn_res = principal_curvature.k_nearest_neighbors(cloud, k)
    curve_res = principal_curvature.principal_curvature(cloud, knn_res)
    curve_idx_reordered = torch.argsort(curve_res)[:desired_num_points]
    return curve_idx_reordered


def bias_curve_fps_sampler(cloud, desired_num_points,k, ratio, bias):
    """Probability based. High bias = more curvature preference vs FPS"""
    if cloud.size(0) < k:
        k = cloud.size(0)

    knn_res = principal_curvature.k_nearest_neighbors(cloud, k)
    curve_res = principal_curvature.principal_curvature(cloud, knn_res)
    # TODO CHECK IF FPS WORKS IF WE PRETEND ITS ALL ONE BATCH
    fps_res= torch_cluster.fps(cloud, torch.ones(size=cloud.shape(0)), ratio, False) # do iterative FPS on all points
    nr_curved = math.ceil(desired_num_points*bias)

    curve_idx_reordered = torch.argsort(curve_res)[:nr_curved]
    # get the rest portion of the points from fps, but only select those we have not selected before
    fps_idx_reordered = fps_res[~fps_res.unsqueeze(1).eq(curve_idx_reordered).any(1)][:desired_num_points-nr_curved]

    return torch.cat([curve_idx_reordered,fps_idx_reordered], 1)





def batch_sampling_coordinator(x, batch, ratio, sampler, sampler_args):
    """
    Coordinate batch-wise point cloud sampling.

    :param x: Tensor of shape (N, C) representing the point cloud data.
    :param batch: Tensor of shape (N,) representing the batch indices of the points.
    :param ratio: Sampling ratio for each point cloud in the batch.
    :param sampler: Sampling algorithm function to be used for sampling points in each point cloud.
    :param sampler_args: Additional arguments to be passed to the sampler function.
    :return: Tensor of shape (M,) containing the indices of the sampled points across the entire batch.

    If `batch` is not None, it should be a tensor of batch indices for each point in `x`. The batch indices
    help identify the point clouds in the batch. It is assumed that `x` and `batch` have compatible shapes,
    with `x` having the same number of points as the length of `batch`.

    The function operates on a batch of point clouds and coordinates the sampling process. It first checks if
    `batch` is provided and performs necessary checks on the sizes and dimensions. It then reshapes `x` into
    individual point clouds. For each point cloud, it calls the provided `sampler` function to sample the
    desired number of points based on the specified `ratio`. The sampled point indices are adjusted to account
    for the position of each point cloud in the batch, and the indices are stored in the output tensor `out`.

    Note: The `sampler` function should take in a point cloud tensor, the desired nr of points, and any additional arguments
    specified in `sampler_args`. It should return the indices of the sampled points within the given point
    cloud.
    """
    if batch is not None:
        assert x.size(0) == batch.numel()
        batch_size = int(batch.max()) + 1

        deg = x.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch, torch.ones_like(batch))

        ptr = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr[1:])
    else:
        ptr = torch.tensor([0, x.size(0)], device=x.device)

    unique_batch = torch.unique(batch)
    num_point_clouds = unique_batch[-1] + 1  # e.g., 32
    nr_points_per_cloud = int(x.size(0) / num_point_clouds)
    x_reshaped = x.view(num_point_clouds, nr_points_per_cloud, -1)  # [32, 40, 3])

    # Iterate over the point clouds
    desired_num_points = math.ceil(ratio * nr_points_per_cloud)
    out = torch.empty(desired_num_points * num_point_clouds, dtype=torch.long)
    for i in range(num_point_clouds):  # use enumerate
        cloud = x_reshaped[i, :, :]

        # compute
        curve_idx_reordered = sampler(cloud, desired_num_points, *sampler_args)

        ptr = i * nr_points_per_cloud  # shift local point index by cloud index
        out[i * desired_num_points:(i + 1) * desired_num_points] = curve_idx_reordered + ptr

    return out


def wrap_curve(x, batch, ratio, k):
    if batch is not None:
        assert x.size(0) == batch.numel()
        batch_size = int(batch.max()) + 1

        deg = x.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch, torch.ones_like(batch))

        ptr = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr[1:])
    else:
        ptr = torch.tensor([0, x.size(0)], device=x.device)
    return by_curvature(x, batch, ratio, k)  # (x, ptr, ratio, k)


def by_curvature(x, batch, ratio, k):
    # TODO, batch is 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4...9,30,30,30,30,31,31,31,31 array
    #  have it precomputed could use it for indexing and paralel compute later
    """
    Select the points based on descending curvature.
    :param x: [points_per_cloud * batchsize, 3]
        Includes all point clouds of p points in all batches


        E.g.,
        4 points for each cloud
        batch = 32: batchsize is 32 clouds per batch, so 4*32 points
        x will then be all clouds in the batch [4*32, 3] tensor
            - need to do FPS only within every 4 indices/ within each point cloud

        Note that points per cloud can change since later layers pool earlier layers. so 4 may go to 2.
         Look at batch for the information.
    :param batch: 1D torch tensor that allocates the points in x to the right cloud
                    e.g., 0,0,0,0,1,1,1,1,2,2,2,2,3,...9,30,30,30,30,31,31,31,31
    :param ratio:
    :param k:
    :return:
    """

    k = 10
    unique_batch = torch.unique(batch)
    num_point_clouds = unique_batch[-1] + 1  # e.g., 32
    nr_points_per_cloud = int(x.size(0) / num_point_clouds)
    x_reshaped = x.view(num_point_clouds, nr_points_per_cloud, -1)  # [32, 40, 3])

    # Iterate over the point clouds
    desired_num_points = math.ceil(ratio * nr_points_per_cloud)
    out = torch.empty(desired_num_points * num_point_clouds, dtype=torch.long)
    for i in range(num_point_clouds):  # use enumerate
        cloud = x_reshaped[i, :, :]
        if cloud.size(0) < k:
            k = cloud.size(0)

        # compute
        knn_res = principal_curvature.k_nearest_neighbors(cloud, k)
        curve_res = principal_curvature.principal_curvature(cloud, knn_res)
        curve_idx_reordered = torch.argsort(curve_res)[:desired_num_points]


        ptr = i * nr_points_per_cloud  # shift local point index by cloud index
        out[i * desired_num_points:(i + 1) * desired_num_points] = curve_idx_reordered + ptr

    return out


def original_fps(x: torch.Tensor, batch: OptTensor = None, ratio: float = 0.5,
                 random_start: bool = True) -> torch.Tensor:
    r"""
    You start with a point cloud comprising N
    points and iteratively select a point until you have up to S
    samples. You have two sets which we will denote sampled and remaining and you choose a point as follows:

    For each point in remaining find its nearest neighbour in sampled, saving the distance.
    Select the point in remaining whose nearest neighbour distance is the largest and move it from remaining to sampled.



    A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    .. code-block:: python

        import torch
        from torch_geometric.nn import fps

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        index = fps(x, batch, ratio=0.5)

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        ratio (float, optional): Sampling ratio. (default: :obj:`0.5`)
        random_start (bool, optional): If set to :obj:`False`, use the first
            node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)

    :rtype: :class:`torch.Tensor`
    """
    return torch_cluster.fps(x, batch, ratio, random_start)
