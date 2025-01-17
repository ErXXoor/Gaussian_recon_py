import torch
from tool import utils
from pytorch3d.ops import knn_points


class Particle:
    def __init__(self, bg_pc):
        self.background_pc = bg_pc

    def init_site_points(self, num_samples):
        sampled_pc = utils.farthest_point_sampling(
            self.background_pc, num_samples)
        normals = utils.estimate_normals(sampled_pc, 0.1)

        self.site_points = torch.tensor(sampled_pc, dtype=torch.float32)
        self.site_normals = torch.tensor(normals, dtype=torch.float32)

    def cal_sigma(self, K=6):
        points = self.site_points.unsqueeze(0)
        knn_result = knn_points(points, points, K=K)
        dists = knn_result.dists[..., 1:]
        dist_sum = dists.sum(dim=-1).unsqueeze(-1)
        aaa = 0

    def __repr__(self):
        return f"Particle(position={self.position}, velocity={self.velocity}, mass={self.mass})"
