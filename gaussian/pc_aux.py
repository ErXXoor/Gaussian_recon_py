import torch
from tool import utils
from pytorch3d.ops import knn_points


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
        self.normals = []
        self.radii = []
        self.init_pc_aux(normals)

        self.background_pc_hd = torch.cat(
            [self.background_pc, self.normals], dim=-1)

    def init_pc_aux(self):
        self.normals = utils.estimate_normals(
            self.background_pc, 0.1).cuda()

        pc_tensor = self.background_pc

        knn_result = knn_points(pc_tensor, pc_tensor, K=6)
        selected_dists = knn_result.dists[..., 5]
        self.radii = 0.75 * selected_dists
        self.radii = self.radii.unsqueeze(-1)

    def init_pc_aux(self, normals):
        self.normals = torch.tensor(
            normals, dtype=torch.float32).unsqueeze(0).cuda()

        pc_tensor = self.background_pc

        knn_result = knn_points(pc_tensor, pc_tensor, K=7)
        selected_dists = knn_result.dists.mean(dim=-1)
        self.radii = (2 * selected_dists).unsqueeze(-1)
