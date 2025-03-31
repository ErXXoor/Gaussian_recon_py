import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import numpy as np


def estimate_tangent_vectors(point_tensor, k=10):
    points = point_tensor.squeeze(0).detach().cpu().numpy()
    dim = points.shape[1]

    eig_0 = np.zeros_like(points)
    eig_1 = np.zeros_like(points)

    neigh = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
    dists, indices = neigh.kneighbors(points)

    for i in range(len(points)):
        neighbor_points = points[indices[i, 1:]]
        pca = PCA(n_components=dim)
        pca.fit(neighbor_points)

        eig_0[i], eig_1[i] = pca.components_[0], pca.components_[1]

    return torch.from_numpy(eig_0).unsqueeze(0).to(point_tensor.device), torch.from_numpy(eig_1).unsqueeze(0).to(point_tensor.device)
