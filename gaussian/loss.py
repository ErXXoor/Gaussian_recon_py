import torch
from pytorch3d.ops import knn_points, knn_gather
from .particle import Particle
import math
from .pc_aux import PC_aux
import torch_scatter
EPS = 1e-10


class Loss_Func:
    def __init__(self):
        pass

    def gauss_energy(self, site_points, ngbr_normals, ngbr_points, sigma):
        dim = site_points.shape[-1]

        site_points_expand = site_points.unsqueeze(-2).expand_as(ngbr_points)

        u_tensor = ngbr_points - site_points_expand

        # u_tensor_dot = torch.einsum(
        #     'abcd,abcd->abc', u_tensor, ngbr_normals)

        # u_tensor = u_tensor - \
        #     u_tensor_dot.unsqueeze(-1)*ngbr_normals

        dist = torch.matmul(u_tensor.unsqueeze(-2),
                            u_tensor.unsqueeze(-1))

        # dist = torch.norm(u_tensor, dim=-1)

        gauss_sigma = 4*sigma*sigma

        e_p = -dist/gauss_sigma
        fgauss = torch.exp(e_p)
        normalization = torch.pow(gauss_sigma*torch.tensor(math.pi), dim/2)
        # fgauss = fgauss / normalization
        fgauss = fgauss.sum(dim=-1)

        return fgauss

    def qem_energy(self, site_points, ngbr_points, ngbr_normals, ngbr_radius, sigma):
        u_tensor = ngbr_points-site_points.unsqueeze(-2).expand_as(ngbr_points)

        dim = u_tensor.shape[-1]
        u_tensor_normal = torch.zeros(dim, dim).expand(
            u_tensor.shape[0], u_tensor.shape[1], u_tensor.shape[2], dim, dim)
        u_tensor_normal = u_tensor_normal.to(u_tensor.device)

        nm_tensor = torch.einsum(
            '...i,...j->...ij', ngbr_normals, ngbr_normals)
        u_tensor_normal[..., :3, :3] = nm_tensor

        u_tensor_normal = torch.einsum(
            "abcde,abcef->abcdf", u_tensor.unsqueeze(-2), u_tensor_normal)
        u_tensor_normal = torch.einsum(
            "abcde,abcef->abc", u_tensor_normal, u_tensor.unsqueeze(-1))

        dist = u_tensor_normal

        nm_sigma = 4*sigma*sigma
        nm_ep = -dist/(nm_sigma)
        normalization = torch.pow(nm_sigma*torch.tensor(math.pi), 1.5)

        f_nm = torch.exp(nm_ep)

        ngbr_radius = ngbr_radius.squeeze(-1)

        # f_nm = f_nm / normalization
        f_nm = f_nm.sum(dim=-1)

        return f_nm

    def qem_energy_vor(self, site_points, bg_points, bg_normals, bg_site_vor, bg_radius, sigma, weight):
        bg_site = knn_gather(
            site_points, bg_site_vor.unsqueeze(-1)).squeeze(-2)
        u_tensor = bg_points-bg_site

        dim = u_tensor.shape[-1]
        u_tensor_normal = torch.zeros(dim, dim).expand(
            u_tensor.shape[0], u_tensor.shape[1], dim, dim)

        u_tensor_normal = u_tensor_normal.to(u_tensor.device)

        nm_tensor = torch.einsum(
            '...i,...j->...ij', bg_normals, bg_normals)
        # nm_tensor = weight * nm_tensor
        u_tensor_normal[..., :3, :3] += nm_tensor

        u_tensor_normal = torch.einsum(
            "abcd,abde->abce", u_tensor.unsqueeze(-2), u_tensor_normal)
        u_tensor_normal = torch.einsum(
            "abcd,abdc->abc", u_tensor_normal, u_tensor.unsqueeze(-1))

        dist = u_tensor_normal

        # sigma = sigma*10
        nm_sigma = 4*sigma*sigma

        nm_ep = -dist/(nm_sigma)
        normalization = torch.pow(nm_sigma*torch.tensor(math.pi), 1.5)

        f_nm = torch.exp(nm_ep)
        # f_nm = f_nm / normalization
        f_nm = f_nm.sum(dim=-1)

        return f_nm

    def cal_loss(self, particles: Particle, epoch, site_K=12, bg_K=20):
        radius = 3*torch.sqrt(torch.tensor(2.0))*particles.sigma

        knn_result_site = knn_points(
            particles.optimize_site_points, particles.optimize_site_points, K=site_K)
        ngbr_points_site = knn_gather(
            particles.optimize_site_points, knn_result_site.idx[:, :, 1:])
        # ngbr_points_site = ngbr_points_site.detach()

        # ngbr_normals_site = knn_gather(
        #     particles.site_normals, knn_result_site.idx[:, :, 1:])
        # ngbr_normals_site = ngbr_normals_site.detach()

        knn_result_bg = knn_points(
            particles.optimize_site_points[..., :3], particles.pc_aux.optimize_base_pc[..., :3], K=bg_K)

        ngbr_points_bg = knn_gather(
            particles.pc_aux.optimize_base_pc, knn_result_bg.idx)
        # ngbr_normals_bg = knn_gather(
        #     particles.pc_aux.normals, knn_result_bg.idx)
        # ngbr_points_bg = ngbr_points_bg.detach()
        # ngbr_normals_bg = ngbr_normals_bg.detach()

        ngbr_radius_bg = knn_gather(
            particles.pc_aux.radii, knn_result_bg.idx)
        # ngbr_radius_bg = ngbr_radius_bg.detach()

        knn_result_bg_vor = knn_points(
            particles.pc_aux.optimize_base_pc[..., :3],
            particles.optimize_site_points[..., :3], K=bg_K)

        ngbr_normals_site = []
        gauss_loss = self.gauss_energy(
            particles.optimize_site_points, ngbr_normals_site, ngbr_points_site,
            particles.sigma)

        # qem_loss = self.qem_energy(
        #     particles.optimize_site_points, ngbr_points_bg, ngbr_normals_bg, ngbr_radius_bg, particles.sigma)

        # qem_loss = self.qem_energy_vor(
        #     particles.optimize_site_points,
        #     particles.pc_aux.optimize_base_pc,
        #     particles.pc_aux.normals, knn_result_bg_vor.idx[..., 0], particles.pc_aux.radii, particles.sigma, weight=decay_qem)

        # decay = 1
        # if epoch % 50 == 0 and epoch != 0:
        #     decay *= 0.8

        # if epoch % 50 == 0 and epoch != 0:
        #     decay *= 0.8

        # loss = [-qem_loss.sum()]

        loss = []
        if particles.dim == 3:
            qem_coeff = 1e-1
            gauss_coeff = 1
            # loss = [-qem_coeff*qem_loss.sum(), gauss_coeff*gauss_loss.sum()]
            loss = [gauss_coeff*gauss_loss.sum()]

        else:
            qem_coeff = 1e3
            gauss_coeff = 1e2
            # loss = [-qem_coeff*qem_loss.sum(), gauss_coeff*gauss_loss.sum()]
            loss = [gauss_loss.sum()]

        return loss
