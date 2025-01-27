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

    ct_site_points = torch.gather(
        fp, 2, min_idx.expand(-1, -1, -1, fp.shape[-1]))

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

    # dist = torch.norm(site_points_expand-proj_site_points,
    #                   dim=-1).unsqueeze(-1)

    # min_dist, min_idx = torch.min(dist, dim=-2, keepdim=True)

    # ct_site_points = torch.gather(
    #     proj_site_points, -2, min_idx.expand(-1, -1, -1, proj_site_points.shape[-1]))
    ct_site_points = proj_site_points[:, :, 0, :].unsqueeze(-2)

    return ct_site_points.squeeze(-2)
