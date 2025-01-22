import open3d as o3d
import numpy as np
import torch


def read_xyz_file(file_path):
    # pcd = o3d.io.read_point_cloud(file_path)
    # return np.asarray(pcd.points)
    data = np.loadtxt(file_path)
    points = data[:, :3]
    normals = data[:, 3:]
    return points, normals


def farthest_point_sampling(point_tensor, num_samples):
    pc = o3d.geometry.PointCloud()
    points = point_tensor.squeeze(0).cpu().numpy()
    pc.points = o3d.utility.Vector3dVector(points)
    samlpes = pc.farthest_point_down_sample(num_samples)

    result_points = np.asarray(samlpes.points, dtype=np.float32)
    return torch.from_numpy(result_points).unsqueeze(0)


def estimate_normals(point_tensor, radius):
    pc = o3d.geometry.PointCloud()
    points = point_tensor.squeeze(0).detach().cpu().numpy()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    result_normals = np.asarray(pc.normals, dtype=np.float32)
    return torch.from_numpy(result_normals).unsqueeze(0)
