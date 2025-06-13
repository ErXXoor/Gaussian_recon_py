import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import robust_laplacian as rbl
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
import fpsample as fps


def gather_neighbor(points, k=12):
    neigh = NearestNeighbors(
        n_neighbors=k+1, radius=0.4, algorithm='ball_tree')
    neigh.fit(points)
    dists, indices = neigh.kneighbors(points, return_distance=True)

    dists_tensor = torch.from_numpy(dists[:, 1:].astype(np.float32))
    indices_tensor = torch.from_numpy(indices[:, 1:].astype(np.int32))
    return dists_tensor, indices_tensor


def disc_project(site_points,
                 bg_points,
                 bg_normals,
                 radius):
    site_points_expand = site_points.unsqueeze(-2).expand_as(bg_points)
    fp = torch.einsum('abcd,abcd->abc', bg_normals,
                      site_points_expand-bg_points).unsqueeze(-1)
    fp = site_points_expand - bg_normals*fp

    dist = torch.norm(fp-bg_points, dim=-1).unsqueeze(-1)

    min_dist, min_idx = torch.min(dist, dim=-2, keepdim=True)

    ct_site_points = torch.gather(
        fp, 2, min_idx.expand(-1, -1, -1, fp.shape[-1]))

    # ct_site_points = fp[:, :, 0, :].unsqueeze(-2)

    return ct_site_points.squeeze(-2)


def disc_project_hd(site_points,
                    bg_points,
                    bg_eg0,
                    bg_eg1,
                    radius):
    site_points_expand = site_points.unsqueeze(-2).expand_as(bg_points)
    site_bg_v = site_points_expand-bg_points
    fp_0 = torch.einsum('abcd,abcd->abc', bg_eg0,
                        site_bg_v).unsqueeze(-1)
    fp_1 = torch.einsum('abcd,abcd->abc', bg_eg1,
                        site_bg_v).unsqueeze(-1)
    proj_v = bg_eg0*fp_0 + bg_eg1*fp_1

    proj_site_points = bg_points + proj_v

    avg_points = proj_site_points[:, :, 0:5, :].sum(dim=-2).unsqueeze(-2)
    avg_points = avg_points / 5
    ct_site_points = avg_points

    # ct_site_points = proj_site_points[:, :, 0, :].unsqueeze(-2)

    return ct_site_points.squeeze(-2)


def disc_project_hd_norm(site_points,
                         bg_points,
                         bg_normals,
                         bg_eg0,
                         bg_eg1,
                         radius):
    site_points_expand = site_points.unsqueeze(-2).expand_as(bg_points)
    site_bg_v = site_points_expand-bg_points

    nm_metric = torch.eye(6).expand(
        site_bg_v.shape[0], site_bg_v.shape[1], site_bg_v.shape[2], 6, 6)
    nm_metric = nm_metric.to(site_bg_v.device)

    normal_tensor = torch.einsum(
        '...i,...j->...ij', bg_normals, bg_normals)
    normal_tensor = normal_tensor
    nm_metric[..., :3, :3] += normal_tensor

    site_bg_v = torch.einsum(
        "abcdd,abcde->abcde", nm_metric, site_bg_v.unsqueeze(-1))
    site_bg_v = site_bg_v.squeeze(-1)

    fp_0 = torch.einsum('abcd,abcd->abc', bg_eg0,
                        site_bg_v).unsqueeze(-1)
    fp_1 = torch.einsum('abcd,abcd->abc', bg_eg1,
                        site_bg_v).unsqueeze(-1)
    proj_v = bg_eg0*fp_0 + bg_eg1*fp_1

    proj_site_points = bg_points + proj_v

    avg_points = proj_site_points[:, :, 0:5, :].sum(dim=-2).unsqueeze(-2)
    avg_points = avg_points / 5
    ct_site_points = avg_points

    # ct_site_points = proj_site_points[:, :, 0, :].unsqueeze(-2)

    return ct_site_points.squeeze(-2)


def diffuse_nn(site_points, K):
    eigen_K = K*2
    diffuse_t = 1.0

    if type(site_points) is torch.Tensor:
        site_point_cpu = site_points.squeeze(0).detach().cpu().numpy()
    else:
        site_point_cpu = site_points

    L, _ = rbl.point_cloud_laplacian(site_point_cpu, n_neighbors=30)
    eigenvals, eigenvecs = eigsh(
        L, k=eigen_K+1, which='SM')

    eigenvals = eigenvals[1:]
    eigenvecs = eigenvecs[:, 1:]

    diffusion_coords = eigenvecs * np.exp(-eigenvals * diffuse_t)
    nbrs = NearestNeighbors(
        n_neighbors=K).fit(diffusion_coords)
    dists, indices = nbrs.kneighbors(diffusion_coords)

    if type(site_points) is torch.Tensor:
        indices_tensor = torch.from_numpy(
            indices).long().unsqueeze(0).to(site_points.device)
    # right now for numpy, do not unsqueeze(0)
    else:
        indices_tensor = indices.astype(np.int32)

    return indices_tensor


def farthest_point_sampling(point_tensor, num_samples):
    points = point_tensor.squeeze(0).cpu().numpy()
    sample_ids = fps.bucket_fps_kdtree_sampling(points, num_samples)

    result_points = torch.from_numpy(points[sample_ids]).unsqueeze(0)
    return result_points, sample_ids


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


def estimate_tangent_vectors(point_tensor, k=9):
    print("Estimating tangent vectors...")
    ##############
    points = point_tensor.squeeze(0).detach().cpu().numpy()
    dim = points.shape[1]

    eig_0 = np.zeros_like(points)
    eig_1 = np.zeros_like(points)

    neigh = NearestNeighbors(
        n_neighbors=k, algorithm='auto').fit(points[..., :3])
    dists, indices = neigh.kneighbors(points[..., :3])

    for i in range(len(points)):
        neighbor_points = points[indices[i]]

        pca = PCA(n_components=dim)
        pca.fit(neighbor_points)

        eig_0[i], eig_1[i] = pca.components_[0], pca.components_[1]

    print("Tangent vectors estimated.")
    return torch.from_numpy(eig_0).unsqueeze(0), torch.from_numpy(eig_1).unsqueeze(0)


def estimate_tangent_vectors_diffuse(point_tensor, k=9):
    print("Estimating tangent vectors...")
    ##############
    points = point_tensor.squeeze(0).detach().cpu().numpy()
    dim = points.shape[1]

    eig_0 = np.zeros_like(points)
    eig_1 = np.zeros_like(points)

    indices = diffuse_nn(
        points[..., :3], k)

    for i in range(len(points)):
        neighbor_points = points[indices[i]]

        pca = PCA(n_components=dim)
        pca.fit(neighbor_points)

        eig_0[i], eig_1[i] = pca.components_[0], pca.components_[1]

    print("Tangent vectors estimated.")
    return torch.from_numpy(eig_0).unsqueeze(0), torch.from_numpy(eig_1).unsqueeze(0)
