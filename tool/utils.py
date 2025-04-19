import open3d as o3d
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def read_xyz_file(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    normals = data[:, 3:]
    return points, normals


def read_xyz_file_8d(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    normals = data[:, 3:6]
    tangents = data[:, 6:8]
    return points, normals, tangents


def farthest_point_sampling(point_tensor, num_samples):
    pc = o3d.geometry.PointCloud()
    points = point_tensor.squeeze(0).cpu().numpy()
    pc.points = o3d.utility.Vector3dVector(points)
    samlpes = pc.farthest_point_down_sample(num_samples)

    result_points = np.asarray(samlpes.points, dtype=np.float32)
    return torch.from_numpy(result_points).unsqueeze(0)


def estimate_normals(point_tensor, k=10):
    points = point_tensor.squeeze(0).detach().cpu().numpy()
    dim = points.shape[1]

    normals = np.zeros_like(points)

    neigh = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
    dists, indices = neigh.kneighbors(points)

    for i in range(len(points)):
        neighbor_points = points[indices[i, 1:]]
        pca = PCA(n_components=dim)
        pca.fit(neighbor_points)

        normals[i] = pca.components_[-1]

    return torch.from_numpy(normals).unsqueeze(0)


def estimate_tangent_vectors(point_tensor, k=30):
    print("Estimating tangent vectors...")
    ##############
    points = point_tensor.squeeze(0).detach().cpu().numpy()
    dim = points.shape[1]

    eig_0 = np.zeros_like(points)
    eig_1 = np.zeros_like(points)

    neigh = NearestNeighbors(
        n_neighbors=k+1, algorithm='auto').fit(points[..., :3])
    dists, indices = neigh.kneighbors(points[..., :3])

    for i in range(len(points)):
        neighbor_points = points[indices[i, 1:]]
        pca = PCA(n_components=dim)
        pca.fit(neighbor_points)

        eig_0[i], eig_1[i] = pca.components_[0], pca.components_[1]

    print("Tangent vectors estimated.")
    return torch.from_numpy(eig_0).unsqueeze(0), torch.from_numpy(eig_1).unsqueeze(0)
