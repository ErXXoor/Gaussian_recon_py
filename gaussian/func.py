import torch
from pytorch3d.ops import knn_points, knn_gather


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

    ct_site_points = torch.gather(fp, 2, min_idx.expand(-1, -1, -1, 3))

    return ct_site_points.squeeze(-2)


def constrained_disc_project(site_points,
                             bg_points,
                             site_normals,
                             radius):
    site_points_expand = site_points.unsqueeze(-2).expand_as(bg_points)
    fp = torch.einsum('abcd,abcd->abc', site_normals,
                      site_points_expand-bg_points).unsqueeze(-1)
    fp = site_points_expand - site_normals*fp

    dist = torch.norm(fp-bg_points, dim=-1).unsqueeze(-1)

    min_fp = bg_points+radius*(fp-bg_points)/dist

    min_dist, min_idx = torch.min(dist, dim=-2, keepdim=True)

    min_site_points = min_fp.gather(-2, min_idx.expand(-1, -1, -1, 3))

    ct_site_points = fp.gather(-2, min_idx.expand(-1, -1, -1, 3))

    mask = min_dist > radius.gather(-2, min_idx)
    ct_site_points = torch.where(mask, min_site_points, ct_site_points)

    return ct_site_points.squeeze(-2)
