import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
import os.path as osp
import torch
import matplotlib.pyplot as plt
import numpy as np

def import_train(nr_points):
    """
    Import training dataset.

    Assumes utils is sitting in the root folder.
    Else add /../utils.py as file.
    :param nr_points:
    :return: torch tensor of training datapoints
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                                    'data/ModelNet10')

    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(nr_points)  # 1024

    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    return train_dataset


def draw_pointcloud(point_cloud, coloring=None, elev=30, azim=340,colorbar=False):
    """

    :param point_cloud: torch.tensor. point_could.pos should be [nr_points,3]
    :param coloring: Color points by. Needs to be same len as nr points.
     If none provided, use x values to improve shading.
    :param elev: Elevation in degrees. Default = 30.
    :param azim: Rotation around the y axis in degrees.
    :return:
    """
    x = point_cloud.pos[:, 0]
    y = point_cloud.pos[:, 1]
    z = point_cloud.pos[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if coloring is None:
        coloring = x
    scatter = ax.scatter(x, y, z, marker='.', alpha=.5, cmap="coolwarm", c=coloring)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    if colorbar:
        plt.colorbar()

def find_neighborhood(points_pos, query_point, radius):
    """
    Return points in some radius of the query point.
    :param points_pos: full point cloud position info. Use point_cloud.pos
    :param query_point: Centroid point.
    :param radius: Radius of neighbourhood.
    :return: torch.tensor of points with shape [N, 3]/
    """
    # Calculate Euclidean distances between query_point and all points
    distances = torch.linalg.vector_norm(points_pos - query_point,dim=1, ord=2)

    # Find indices of points within the specified radius
    neighborhood_indices = torch.where(distances <= radius)[0]

    # Return the points in the neighborhood
    neighborhood_points = points_pos[neighborhood_indices]

    return neighborhood_points


def draw_pointcloud_neighbour(points_pos, neighbours, center, radius,limit_axis, elev=30, azim=260):
    """
    Draws a 3D plot of the pointcloud highlighting the region identified by the neighbourhood function.
    :param points_pos: point_cloud.pos torch tensor
    :param neighbours: neighbouring points from the find_neighbours function.
    :param center: center point
    :param radius: radius used
    :param limit_axis: Bool. If should try to plot only the local region around the center point.
    :param elev: Elevation in degrees. Default = 30.
    :param azim: Rotation around the y axis in degrees.
    :return:
    """

    x = points_pos[:, 0]
    y = points_pos[:, 1]
    z = points_pos[:, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the points
    ax.scatter(x, y, z, c='b', marker='.', alpha=0.01)
    ax.scatter(neighbours[:, 0], neighbours[:, 1], neighbours[:, 2], c='red', marker=".")
    ax.scatter(center[0], center[1], center[2], c='cyan', marker="*", s=80)

    # plot only around radius
    x_min, x_max = center[0] - radius, center[0] + radius
    y_min, y_max = center[1] - radius, center[1] + radius
    z_min, z_max = center[2] - radius, center[2] + radius
    if limit_axis:
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)

def visualize_plane(center_point, surrounding_points,point_cloud, radius, limit_axis=False,elev=30,azim=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for surrounding points
    ax.scatter(surrounding_points[:, 0], surrounding_points[:, 1], surrounding_points[:, 2], c='orange', label='Surrounding Points', alpha=0.1)

    # Scatter plot for center point
    ax.scatter(center_point[0], center_point[1], center_point[2], c='r', marker='o', label='Center Point',s=100)

    # Scatter plot for point_cloud
    x = point_cloud.pos[:, 0]
    y = point_cloud.pos[:, 1]
    z = point_cloud.pos[:, 2]

    ax.scatter(x, y, z, marker='.', alpha=.01)

    # Plane
    v1 = surrounding_points[1] - surrounding_points[0]
    v2 = surrounding_points[2] - surrounding_points[0]
    normal_vector = np.cross(v1, v2)
    d = -np.dot(normal_vector, surrounding_points[0])
    x_range = np.linspace(center_point[0]-radius, center_point[1]+radius, 10)
    y_range = np.linspace(center_point[1]-radius, center_point[1]+radius, 10)
    x_plane, y_plane = np.meshgrid(x_range, y_range)
    z_plane = (-normal_vector[0] * x_plane - normal_vector[1] * y_plane - d) / normal_vector[2]

    # Plot the plane
    ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.3, color='g', label='Plane')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # legend
    scatter_proxy = plt.Line2D([0], [0], linestyle='none', c='b', marker='o')
    plane_proxy = plt.Line2D([0], [0], linestyle='-', c='g')
    center_proxy = plt.Line2D([0], [0], linestyle='none', c='r', marker='o')
    ax.legend([scatter_proxy, plane_proxy, center_proxy], ['Surrounding Points', 'Plane', 'Center Point'])
    ax.view_init(elev=elev, azim=azim)
    x_min, x_max = center_point[0] - radius, center_point[0] + radius
    y_min, y_max = center_point[1] - radius, center_point[1] + radius
    z_min, z_max = center_point[2] - radius, center_point[2] + radius
    if limit_axis:
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])





