from torch_geometric.data import Data
import torch
import sampling_algs
import principal_curvature

def nest_idxsampler(pos, idx_sampler, nr_points, sampling_args, cloud_idx=None):
    """
    The sampling alg functions return the selected indices of the points.
     This function uses one of these samplers, applies it to a pointcloud position data and returns the originaal [nr_points,3] format.
    :param pos:
    :param idx_sampler:
    :param nr_points:
    :param sampling_args:
    :return:
    """
    if cloud_idx is not None:
        selected_indices = idx_sampler(pos, nr_points, cloud_idx, *sampling_args)
    else:
        selected_indices = idx_sampler(pos, nr_points, *sampling_args)
    return pos[selected_indices, :]


def apply_subsample_transform(modelnet_data, sampler, nr_points, sampling_args=[], sampler_type="idx", pass_idx=False):
    """
    Apply a preprocessing sampler to the dataset and return the new dataset for the dataloader.
    :param modelnet_data:
    :param sampler: Sample pointcloud:
            input->output: pointcloud.pos -> pointcloud.pos
            arg
    :return:
    """

    transformed_data = list()
    for idx, cloud in enumerate(modelnet_data):
        args = []
        if sampler_type == "idx":
            if pass_idx:
                new_pos = nest_idxsampler(cloud.pos, sampler, nr_points, sampling_args,
                                          cloud_idx=idx)  # returns torch tensor of shape [nr points, 3]
            else:
                new_pos = nest_idxsampler(cloud.pos, sampler, nr_points,
                                          sampling_args)  # returns torch tensor of shape [nr points, 3]

        else:
            new_pos = sampler(cloud.pos, nr_points, *sampling_args)
        new_data = Data(pos=new_pos, y=cloud.y)
        transformed_data.append(new_data)
    return transformed_data



def get_minimal_above_average_set(data, curvature_values):
    """
    # the minimum number of above average points across clouds. Since FPS needs all clouds to be the same size,
    we nr_points-this_minimum number of least curved points from all clouds.

    # Note. We know that the largest N points are above average curvature for all point clouds,
     since we choose N based on the smallest number of point that are above average curvature in any point cloud.
      Since this is met for the smallest point cloud, it is met for the larger ones by induction.
    :param data:
    :param curvature_values:
    :return:
        out_data:python list of pointclouds
        new_nr_points: the new size of the pointcloud

    """
    data = data.clone()
    above_average_mask = data > torch.mean(curvature_values, axis=0)
    new_nr_points = above_average_mask.int().sum(axis=0).min()  # must be ≥ the nr of points we want in the NN
    out_data = [None] * data.len()
    for i, cloud in enumerate(data):
        # select the largest number of points that are above average curvature
        selected_indices = torch.argsort(curvature_values[:, i], descending=True)[:new_nr_points]
        out_data[i] = cloud.pos[selected_indices, :]
    return out_data, new_nr_points


def preprocess(data, nr_points, method="fps",nr_candidates = 10,):
    if method =="fps":
        return apply_subsample_transform(data, sampling_algs.fps_pure, nr_points, sampler_type="idx")
    elif method == "biased_fps":
        curvature_values = principal_curvature.get_curvatures(data)
        return apply_subsample_transform(data, sampling_algs.fps_top_n_v2, nr_points,
                                                            sampling_args=[curvature_values, nr_candidates], pass_idx=True)
    elif method == "above_avg_curvature":
        curvature_values = principal_curvature.get_curvatures(data)
        new_test_data, new_nr_points = get_minimal_above_average_set(data, curvature_values)
        assert new_nr_points >= nr_points, "Need to sample more points from model net. Nr above average points too small."
        return apply_subsample_transform(data, sampling_algs.fps_pure, nr_points, sampler_type="idx")


# ----Usage----
# # Baseline:
# test_dataset_fps = apply_subsample_transform(test_dataset, sampling_algs.fps_pure, nr_points, sampler_type="idx")

# # Biased FPS (candidates)
# nr_candidates = 10
# curvature_values = principal_curvature.get_curvatures(test_dataset)
# test_dataset_biased_fps = apply_subsample_transform(test_dataset, sampling_algs.fps_top_n_v2, nr_points,
#                                                     sampling_args=[curvature_values, nr_candidates], pass_idx=True)

# FPS on above average curved points
# curvature_values = principal_curvature.get_curvatures(test_dataset)
# new_test_data, new_nr_points = get_minimal_above_average_set(test_dataset, curvature_values)
# assert new_nr_points >= nr_points, "Need to sample more points from model net. Nr above average points too small."
# test_dataset_above_avg = apply_subsample_transform(test_dataset, sampling_algs.fps_pure, nr_points, sampler_type="idx")
