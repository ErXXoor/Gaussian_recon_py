import numpy as np
from tool.utils import estimate_normals
import torch
from rvd.RVD import rvd_rec


def read_xyz_normal_file(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    normals = data[:, 3:]
    return points, normals


def normalize(points):
    min_coord = points.min(axis=0)
    max_coord = points.max(axis=0)
    center = (min_coord + max_coord) / 2
    scale = max(max_coord - min_coord)
    points = (points - center) / scale
    return points


if __name__ == "__main__":
    xyz_file = "/home/hongbo/Desktop/code/Freedman/tmp/9ae3aa0.xyz"
    output_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/data/9ae3aa0.xyz"

    points, normals = read_xyz_normal_file(xyz_file)
    normal_points = normalize(points)

    point_data = np.hstack((normal_points, normals))
    np.savetxt(xyz_file, point_data, fmt="%.6f")
