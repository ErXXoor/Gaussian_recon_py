import torch
from tool import utils
from pytorch3d.ops import knn_points, knn_gather
from .pc_aux import PC_aux
from .func import constrained_disc_project, disc_project


class Particle:
    def __init__(self, pc_aux: PC_aux, site_num=10000):
        self.pc_aux = pc_aux
        self.area = 0.0
        self.sigma = 0.0

        self.init_site_points(site_num)
        self.cal_sigma()

    def init_site_points(self, num_samples):
        self.site_points = utils.farthest_point_sampling(
            self.pc_aux.background_pc, num_samples)

        self.site_points = self.site_points.cuda()
        self.site_points.requires_grad = True

        self.update_normals()

        self.optimize_site_points = torch.cat(
            [self.site_points, self.site_normals], dim=-1)
        self.optimize_site_points.requires_grad = True

    def cal_sigma(self, K=6):
        with torch.no_grad():
            knn_result = knn_points(self.site_points, self.site_points, K=K)
            dists = knn_result.dists[..., 1:]
            dist_sum = dists.sum(dim=-1)

            self.area = 0.86/25 * torch.pow(dist_sum, 2).sum()
            self.sigma = 0.32*torch.sqrt(self.area/len(self.site_points))

    def update_normals(self):
        with torch.no_grad():
            knn_result = knn_points(
                self.site_points, self.pc_aux.background_pc, K=6)
            bg_normals = knn_gather(
                self.pc_aux.normals, knn_result.idx[..., 0:1]).squeeze(-2)
            self.site_normals = bg_normals

    def constrain_sites(self, K=20):
        with torch.no_grad():
            bg_points_tensor = self.pc_aux.background_pc
            knn_result = knn_points(self.site_points,
                                    bg_points_tensor,
                                    K=K)
            normals = knn_gather(self.pc_aux.normals,
                                 knn_result.idx)
            points = knn_gather(bg_points_tensor,
                                knn_result.idx)
            radii = knn_gather(self.pc_aux.radii,
                               knn_result.idx)
            new_points = disc_project(
                self.site_points,
                points,
                normals,
                radii)
            self.site_points += new_points - self.site_points

    def constrain_sites_final(self, K=20):
        with torch.no_grad():
            bg_points_tensor = self.pc_aux.background_pc
            knn_result = knn_points(self.site_points,
                                    bg_points_tensor,
                                    K=K)
            normals = knn_gather(self.pc_aux.normals,
                                 knn_result.idx)
            points = knn_gather(bg_points_tensor,
                                knn_result.idx)
            radii = knn_gather(self.pc_aux.radii,
                               knn_result.idx)
            new_points = constrained_disc_project(
                self.site_points,
                points,
                normals,
                radii)
            self.site_points += new_points - self.site_points
