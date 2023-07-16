"""
Default call returns a 1D torch tensor of indices for the points in the sample.
Could be that dim is higher if multiple batches.
Call: idx = fps(pos, batch, ratio=self.ratio)


"""

import torch
import torch_cluster

from torch_geometric.typing import OptTensor
import principal_curvature
import scipy.stats as stats
import math


# def interpolate_with_distribution(distribution, z_score):
#     # Calculate the left and right bounds based on the distribution
#     # Calculate the bias based on the z-score
#     # Perform linear interpolation
#
#
#     left_quantile = quantile(distribution, 0.025)
#     right_quantile = quantile(distribution, 0.975)
#
#     bias = stats.norm.cdf(z_score)
#     result = left_quantile + (right_quantile - left_quantile) * bias
#     return result



# def principal_curvature_fps(x, batch, ratio, z_score_bias,k=5):
#     """
#     Sample more strongly from curved spaces.
#     FPS but we reduce possible points that can be sampled based on the curvature.
#     :param x:
#     :param batch:
#     :param ratio:
#     :param z_score_bias:
#     :param k:
#     :return:
#     """
#
#     knn_res = principal_curvature.k_nearest_neighbors(x, k)
#     curve_res = principal_curvature.principal_curvature(x, knn_res)
#
#     # calculating the threshold based on the z-score
#     # is likely expensive, we could also opt for a more simple min max bias
#     # threshold = interpolate_with_distribution(curve_res, z_score_bias)
#     # x = x[curve_res > threshold]
#
#     # fps wants ratio to return a certain ratio of the original x
#     # now we need to ensure that this ratio is still intact after our bias
#
#     x_reduced = x[curve_res > mean(curve_res)]# !! would need to make sure that x_reduced is at least nr desired points long
#
#
#     desired_num_points = int(ratio * x.size(0))
#     print("here", desired_num_points)
#     # Calculate the reduction factor based on the number of points in `y` and `x`
#     # reduction_factor = float(x_reduced.size(0)) / float(x.size(0))
#     # adjusted_ratio = math.ceil(ratio / reduction_factor, 2)
#
#     batch = batch[curve_res > mean(curve_res)]
#     # print(batch)
#     # print(batch.shape)
#
#     sampled_idx = torch_cluster.fps(x_reduced, batch, ratio=1.0)
#     print("sampled_idx",sampled_idx)
#     adjusted_indices = x_reduced[:desired_num_points]
#     # adjusted_indices here are of x_reduced, but we need them to be indices from x selected out of x_reduced!
#     print("batch shape", batch.shape)
#     print("x shape", x_reduced.shape)
#     print("len adj idx", len(adjusted_indices))
#
#
#
#     return adjusted_indices

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
    return by_curvature(x, batch, ratio, k) # (x, ptr, ratio, k)


def by_curvature(x, batch, ratio,k):
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
    k = 0 # FIXME must check if at least k+1 points are in the cloud
    total_nr_points = batch.size(0)
    unique_batch = torch.unique(batch)
    num_point_clouds = unique_batch[-1]+1  # e.g., 32
    nr_points_per_cloud = int(x.size(0) / num_point_clouds)
    # print("nr_points_per_cloud, x.size(0), num_point_clouds: ", nr_points_per_cloud, x.size(0), num_point_clouds)

    # assert nr_points_per_cloud * num_point_clouds == total_nr_points, "Not all point clouds have the same nr of points."

    x_reshaped = x.view(num_point_clouds, nr_points_per_cloud, -1)  # [32, 40, 3])

    # Iterate over the point clouds
    desired_num_points = math.ceil(ratio * nr_points_per_cloud)
    out = torch.empty(desired_num_points * num_point_clouds, dtype=torch.long)
    for i in range(num_point_clouds):  # use enumerate

        cloud = x_reshaped[i,:,:]
        # compute
        # knn_res = principal_curvature.k_nearest_neighbors(cloud, k)
        # curve_res = principal_curvature.principal_curvature(cloud, knn_res)
        # curve_idx_reordered = torch.argsort(curve_res)[:desired_num_points]
        # print("curve res, curve idx reordered", curve_res, curve_idx_reordered)

        curve_idx_reordered = torch.arange(cloud.size(0))[:desired_num_points]  # dummy
        # print("cloud shape", cloud.size(0))
        # print("curve_idx_reordered.shape,curve_idx_reordered ,  desired_num_points", curve_idx_reordered.shape, curve_idx_reordered, desired_num_points)
        ptr = i * nr_points_per_cloud # shift local point index by cloud index
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