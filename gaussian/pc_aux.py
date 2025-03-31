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
        self.init_pc_aux()

    def __init__(self, bg_pc, normals):
        self.background_pc = torch.tensor(
            bg_pc, dtype=torch.float32).unsqueeze(0).cuda()
        self.normals = torch.tensor(
            normals, dtype=torch.float32).unsqueeze(0).cuda()
        self.radii = []

        # self.optimize_base_pc = torch.cat(
        #     [self.background_pc, 0.3*self.normals], dim=-1)

        self.optimize_base_pc = self.background_pc

        # self.hd_eig0, self.hd_eig1 = utils.estimate_tangent_vectors(
        #     self.optimize_base_pc)
        # self.hd_eig0 = self.hd_eig0.cuda()
        # self.hd_eig1 = self.hd_eig1.cuda()
        self.init_radii()

    def init_radii(self):
        pc_tensor = self.optimize_base_pc
        knn_result = knn_points(pc_tensor, pc_tensor, K=7)
        selected_dists = knn_result.dists.mean(dim=-1)

        # pc_tensor = torch.cat([self.background_pc, self.normals], dim=-1)
        # knn_result = knn_points(self.optimize_base_pc,
        #                         self.optimize_base_pc, K=7)

        # pc_tensor_ngbr = knn_gather(pc_tensor, knn_result.idx[..., 1:])
        # pc_tensor_expand = pc_tensor.unsqueeze(-2).expand_as(pc_tensor_ngbr)

        # selected_dists = torch.norm(
        #     pc_tensor_ngbr-pc_tensor_expand, dim=-1).mean(dim=-1)

        self.radii = math.pi*selected_dists**2
        self.radii = self.radii.unsqueeze(-1)
