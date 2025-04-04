import torch
from tool import utils
from pytorch3d.ops import knn_points, knn_gather
import math


class PC_aux:
    def __init__(self, bg_pc):
        self.background_pc = torch.tensor(
            bg_pc, dtype=torch.float32).unsqueeze(0).cuda()
        self.normals = []
        self.radii = []

    def __init__(self, bg_pc, normals):
        self.background_pc = torch.tensor(
            bg_pc, dtype=torch.float32).unsqueeze(0).cuda()
        self.normals = torch.tensor(
            normals, dtype=torch.float32).unsqueeze(0).cuda()
        self.radii = []

        self.optimize_base_pc = torch.cat(
            [self.background_pc, 0.1*self.normals], dim=-1)

        # self.optimize_base_pc = self.background_pc

        self.hd_eig0, self.hd_eig1 = utils.estimate_tangent_vectors(
            self.optimize_base_pc)
        self.hd_eig0 = self.hd_eig0.cuda()
        self.hd_eig1 = self.hd_eig1.cuda()
        self.init_radii()

    def init_radii(self):
        pc_tensor = self.optimize_base_pc
        knn_result = knn_points(pc_tensor, pc_tensor, K=7)
        selected_dists = knn_result.dists.mean(dim=-1)

        selected_dists = selected_dists

        self.radii = 1.3*math.pi*selected_dists**2

        # self.radii = math.pi*selected_dists**2
        self.radii = self.radii.unsqueeze(-1)

    def get_total_area(self):
        return self.radii.sum()
