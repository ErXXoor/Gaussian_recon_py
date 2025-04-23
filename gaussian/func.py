import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np


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
