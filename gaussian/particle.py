import torch
from tool import utils
from pytorch3d.ops import knn_points, knn_gather
from .pc_aux import PC_aux
from .func import disc_project, disc_project_hd, disc_project_hd_norm


class Particle:
    def __init__(self, pc_aux: PC_aux, site_num=10000, dim=3):
        self.pc_aux = pc_aux
        self.area = 0.0
        self.sigma = 0.0
        self.dim = dim

        self.init_site_points(site_num)
        # self.cal_sigma()
        self.update_sigma()

    def init_site_points(self, num_samples):
        self.site_points = utils.farthest_point_sampling(
            self.pc_aux.background_pc, num_samples)

        self.site_points = self.site_points.cuda()

        self.update_normals()

        if self.dim == 3:
            self.optimize_site_points = self.site_points

        elif self.dim == 6:
            self.optimize_site_points = torch.cat(
                [self.site_points, 0.1*self.site_normals], dim=-1)

        self.optimize_site_points.requires_grad = True

    def cal_sigma(self, K=6):
        area = self.pc_aux.get_total_area()
        with torch.no_grad():
            # knn_result = knn_points(
            #     self.optimize_site_points, self.optimize_site_points, K=K)
            # dists = knn_result.dists[..., 1:]
            # dist_sum = dists.sum(dim=-1)

            # self.area = 0.86/25 * torch.pow(dist_sum, 2).sum()
            alpha = 0.32

            self.area = 0.86/25 * area
            # alpha = 1

            self.sigma = alpha * \
                torch.sqrt(self.area/len(self.optimize_site_points))

    def update_sigma(self, K=7):
        with torch.no_grad():
            knn_result = knn_points(
                self.optimize_site_points, self.optimize_site_points, K=K)
            dists = knn_result.dists[..., 1:]
            dist_sum = dists.mean(dim=-1)

            aaa = torch.pow(dist_sum, 2).sum()

            self.area = 0.86/25 * torch.pow(dist_sum, 2).sum()

            self.sigma = 1 * \
                torch.sqrt(self.area/len(self.optimize_site_points))

            bbb = 0

    def update_normals(self):
        with torch.no_grad():
            knn_result = knn_points(
                self.site_points, self.pc_aux.background_pc, K=6)
            bg_normals = knn_gather(
                self.pc_aux.normals, knn_result.idx[..., 0:1]).squeeze(-2)
            self.site_normals = bg_normals

    def update_site_points(self):
        self.site_points = self.optimize_site_points[..., :3]

    def constrain_sites(self, K=20):
        with torch.no_grad():
            bg_points_tensor = self.pc_aux.optimize_base_pc
            knn_result = knn_points(self.optimize_site_points,
                                    bg_points_tensor,
                                    K=K)
            normals = knn_gather(self.pc_aux.normals,
                                 knn_result.idx)
            points = knn_gather(bg_points_tensor,
                                knn_result.idx)
            radii = knn_gather(self.pc_aux.radii,
                               knn_result.idx)

            new_points = disc_project(
                self.optimize_site_points,
                points,
                normals,
                radii)
            self.optimize_site_points += new_points - self.optimize_site_points

    def constrain_sites_hd(self, K=7):
        with torch.no_grad():
            bg_points_tensor = self.pc_aux.optimize_base_pc

            # knn_result = knn_points(self.optimize_site_points,
            #                         bg_points_tensor,
            #                         K=K)

            knn_result = knn_points(self.optimize_site_points[..., :3],
                                    bg_points_tensor[..., :3],
                                    K=K)

            bg_eig0 = knn_gather(self.pc_aux.hd_eig0,
                                 knn_result.idx)
            bg_eig1 = knn_gather(self.pc_aux.hd_eig1,
                                 knn_result.idx)
            bg_points = knn_gather(bg_points_tensor,
                                   knn_result.idx)
            radii = knn_gather(self.pc_aux.radii,
                               knn_result.idx)
            new_points = disc_project_hd(
                self.optimize_site_points,
                bg_points,
                bg_eig0,
                bg_eig1,
                radii)

            # # metric projection
            # bg_normals = knn_gather(
            #     self.pc_aux.normals, knn_result.idx).squeeze(-2)

            # new_points = disc_project_hd_norm(
            #     self.optimize_site_points,
            #     bg_points,
            #     bg_normals,
            #     bg_eig0,
            #     bg_eig1,
            #     radii)

            self.optimize_site_points += new_points - self.optimize_site_points
