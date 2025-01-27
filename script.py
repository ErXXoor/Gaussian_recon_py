import numpy as np
from tool.utils import estimate_normals
import torch
from rvd.RVD import rvd_rec


def read_xyz_normal_file(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    normals = data[:, 3:]
    return points, normals


if __name__ == "__main__":
    xyz_file = "/home/hongbo/Desktop/code/Gaussian_recon_py/data/guitar.xyz"
    output_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/test.xyz"

    points, normals = read_xyz_normal_file(xyz_file)
    points = torch.from_numpy(points).unsqueeze(0)
    normals = torch.from_numpy(normals).unsqueeze(0)

    rvd_rec(points, normals)
