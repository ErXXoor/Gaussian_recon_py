import torch
from pytorch3d.ops import knn_points, knn_gather
from .particle import Particle
import math
from .pc_aux import PC_aux


class Loss_Func:
    def __init__(self):
        pass

    def gauss_energy(self, site_points, ngbr_normals, ngbr_points, sigma):
        site_points_expand = site_points.unsqueeze(-2).expand_as(ngbr_points)

        u_tensor = ngbr_points - site_points_expand

        # u_tensor_dot = torch.einsum(
        #     'abcd,abcd->abc', u_tensor, ngbr_normals)

        # u_tensor = u_tensor - \
        #     u_tensor_dot.unsqueeze(-1)*ngbr_normals

        # dist = torch.matmul(u_tensor.unsqueeze(-2),
        #                     u_tensor.unsqueeze(-1))

        dist = torch.norm(u_tensor, dim=-1)
        aaa = dist.detach().cpu().numpy()

        e_p = -dist/(4*sigma*sigma)
        fgauss = torch.exp(e_p)
        normalization = (sigma*sigma*sigma*math.pow(2*math.pi, 1.5))**2
        bbb = fgauss.detach().cpu().numpy()
        fgauss = fgauss / normalization
        fgauss = fgauss.sum(dim=-1)

        return fgauss

    def qem_energy(self, site_points, ngbr_points, ngbr_normals, ngbr_radius, sigma):
        u_tensor = ngbr_points-site_points.unsqueeze(-2).expand_as(ngbr_points)

        # radius_mask = torch.norm(u_tensor, dim=-1) < 0.3*sigma
        # u_tensor = u_tensor[radius_mask]
        # ngbr_normals = ngbr_normals[radius_mask]
        # u_tensor = torch.einsum('ab,ab->a', u_tensor, ngbr_normals)

        u_tensor = torch.einsum('abcd,abcd->abc', u_tensor, ngbr_normals)

        dist = u_tensor**2
        nm_ep = -dist/(4*sigma*sigma)
        f_nm = torch.exp(nm_ep)
        normalization = (sigma*sigma*sigma*math.pow(2*math.pi, 1.5))**2
        f_nm = f_nm / normalization
        f_nm = f_nm.sum(dim=-1)

        return f_nm

    def cal_loss(self, particles: Particle, site_K=10, bg_K=20):
        radius = 3*torch.sqrt(torch.tensor(2.0))*particles.sigma
        knn_result_site = knn_points(
            particles.site_points, particles.site_points, K=site_K)
        ngbr_points_site = knn_gather(
            particles.site_points, knn_result_site.idx[:, :, 1:])

        ngbr_points_site = ngbr_points_site.detach()
        ngbr_normals_site = knn_gather(
            particles.site_normals, knn_result_site.idx[:, :, 1:])
        ngbr_normals_site = ngbr_normals_site.detach()

        # site_6d = torch.concat((particles.site_points,
        #                        0.1*particles.site_normals), dim=-1)
        # bg_6d = torch.concat((particles.pc_aux.background_pc,
        #                      0.1*particles.pc_aux.normals), dim=-1)
        # knn_result_bg = knn_points(site_6d, bg_6d, K=bg_K)

        knn_result_bg = knn_points(
            particles.site_points, particles.pc_aux.background_pc, K=bg_K)

        ngbr_points_bg = knn_gather(
            particles.pc_aux.background_pc, knn_result_bg.idx)
        ngbr_normals_bg = knn_gather(
            particles.pc_aux.normals, knn_result_bg.idx)
        ngbr_points_bg = ngbr_points_bg.detach()
        ngbr_normals_bg = ngbr_normals_bg.detach()

        ngbr_radius_bg = knn_gather(
            particles.pc_aux.radii, knn_result_bg.idx)
        ngbr_radius_bg = ngbr_radius_bg.detach()

        gauss_loss = self.gauss_energy(
            particles.site_points, ngbr_normals_site, ngbr_points_site, radius)
        qem_loss = self.qem_energy(
            particles.site_points, ngbr_points_bg, ngbr_normals_bg, ngbr_radius_bg, radius)

        # loss = [gauss_loss.sum()]
        loss = [-1e3*qem_loss.sum(), gauss_loss.sum()]
        return loss
