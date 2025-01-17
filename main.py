from tool import utils, vis
import numpy as np
import open3d as o3d
from gaussian.particle import Particle

if __name__ == "__main__":
    xyz_path = "/Users/lihongbo/Desktop/code/Gauss_recon_py/data/01_82-block.xyz"
    site_num = 10000

    point_cloud = utils.read_xyz_file(xyz_path)

    particles = Particle(point_cloud)
    particles.init_site_points(site_num)
    particles.cal_sigma()
