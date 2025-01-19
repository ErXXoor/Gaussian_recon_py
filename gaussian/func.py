import torch
from pytorch3d.ops import knn_points


def constrained_disc_project(site_points,
                             bg_points,
                             site_normals,
                             radius):
    site_points_expand = site_points.unsqueeze(-2).expand_as(bg_points)
    fp = torch.einsum('abcd,abcd->abc', site_normals,
                      site_points_expand-bg_points).unsqueeze(-1)
    fp = site_points_expand - site_normals*fp

    dist = torch.norm(fp-bg_points, dim=-1).unsqueeze(-1)

    min_dist, min_idx = torch.min(dist, dim=-2, keepdim=True)

    ct_site_points = fp.gather(-2, min_idx.expand(-1, -1, -1, 3))
    # mask = min_dist > radius.gather(-2, min_idx)

    # if mask.any():
    #     ct_dir = fp.gather(-2, min_idx)[mask] - \
    #         bg_points.gather(-2, min_idx)[mask]
    #     ct_dir = ct_dir / torch.norm(ct_dir, dim=-1, keepdim=True)
    #     ct_site_points[mask] = bg_points.gather(
    #         -2, min_idx)[mask] + ct_dir * radius.gather(-2, min_idx)[mask]

    return ct_site_points.squeeze(-2)
