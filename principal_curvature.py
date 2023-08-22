import torch

def k_nearest_neighbors(point_cloud_pos, k):
    """
    Find indices of kNN for every point in the point cloud.
    :param point_cloud_pos: Torch tensor of shape [N, 3] representing a point cloud, where N is the number of points and each point is represented by 3D coordinates (x, y, z).
    :param k: Integer value representing the number of nearest neighbors to retrieve for each point in the cloud.
    :return: Torch tensor of shape [N, k] containing the indices of the k nearest neighbors for each point in the point cloud.
    """
    query_points = point_cloud_pos.unsqueeze(1)
    distances = torch.linalg.vector_norm(point_cloud_pos - query_points, dim=2, ord=2)
    _, indices = torch.topk(distances, k=k+1, largest=False) # k+1 bc point_cloud has self too

    return indices

def covariance_matrix(point_cloud_pos, knn_idx):
    """
    Compute the covariance matrix of a point cloud.

    :param point_cloud_pos: torch tensor of size [N,3] representing the positions of points in the point cloud.
    :param knn_idx: torch tensor [N,k+1] containing the indices of each point and its k nearest neighbors.
    :return: The covariance matrix of the point cloud.
    """
    knn = point_cloud_pos[knn_idx]
    diffs = knn - knn.mean(1, keepdims=True)
    reorder_diffs = torch.transpose(diffs, 1, 2) # 0, 2, 1
    return torch.matmul(reorder_diffs, diffs) / knn.shape[1]


def compute_eigenvalues(point_cloud_pos, knn_idx):
    """
    Calculates the eigenvalues of the covariance matrix for a given point cloud.
    Eigenvalue columns are sorted from smallest to largest.

    :param point_cloud_pos: Tensor of shape (N, 3) representing the positions of the points in the point cloud,
                            where N is the number of points and D=3 is the dimensionality of the points.
    :param knn_idx: Tensor of shape (N, K) containing the indices of the K nearest neighbors for each point
                    in the point cloud.
    :return: Tensor of shape (N, 3) containing the top three eigenvalues of the covariance matrix for each point
             in the point cloud, stacked column-wise.
    """
    cov = covariance_matrix(point_cloud_pos, knn_idx)
    eigenvalues = torch.linalg.eigvals(cov).to(torch.float)

    order = torch.argsort(eigenvalues,axis=1,descending=True)

    return torch.column_stack((eigenvalues[range(len(eigenvalues)), order[:, 0]],
                        eigenvalues[range(len(eigenvalues)), order[:, 1]],
                        eigenvalues[range(len(eigenvalues)), order[:, 2]]))


def principal_curvature(point_cloud_pos, neighbor_idx):
    """
    Compute principal curvature based on the eigenvalues derived of some neighbourhood (e.g., 3-NN) around each point.

    :param point_cloud_pos: point_cloud.pos [N,3] torch tensor of coordinates.
    :param neighbor_idx: index data of the neighbourhood of each point. Output of e.g., k_nearest_neighbors.
    :return:
    """
    eigenv = compute_eigenvalues(point_cloud_pos,neighbor_idx)
    denominator = eigenv[:, 0] + eigenv[:, 1] + eigenv[:, 2]
    result = torch.zeros(len(denominator), device=point_cloud_pos.device) #.to(cloud.device)
    nonzero_indices = denominator != 0
    result[nonzero_indices] = eigenv[:, 2][nonzero_indices] / denominator[nonzero_indices]
    return result


def curvatures_knn(point_cloud_pos, k):
    """
    Return curvature values for the point cloud.
    :param point_cloud_pos: point_cloud.pos [N,3] torch tensor of coordinates.
    :param k: nr of knn
    :return: 1D torch.tensor for curvature values. Indices match the ones of the pointcloud.
    """
    if point_cloud_pos.size(0) < k:
        k = point_cloud_pos.size(0)

    knn_res = k_nearest_neighbors(point_cloud_pos, k)
    return principal_curvature(point_cloud_pos, knn_res)





