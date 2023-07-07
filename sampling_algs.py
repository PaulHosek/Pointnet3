"""
Default call returns a 1D torch tensor of indices for the points in the sample.
Could be that dim is higher if multiple batches.
Call: idx = fps(pos, batch, ratio=self.ratio)


"""

from torch import Tensor
import torch_cluster

from torch_geometric.typing import OptTensor


def curvature_bias():
    """
    Sample more strongly from curved spaces.

    :return:
    """



def original_fps(x: Tensor, batch: OptTensor = None, ratio: float = 0.5,
        random_start: bool = True) -> Tensor:
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