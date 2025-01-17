import open3d as o3d
import numpy as np


def read_xyz_file(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def farthest_point_sampling(points, num_samples):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    samlpes = pc.farthest_point_down_sample(num_samples)
    return np.asarray(samlpes.points)


def estimate_normals(points, radius):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return np.asarray(pc.normals)
