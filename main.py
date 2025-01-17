from tool import utils, vis
import numpy as np
import polyscope as ps
import open3d as o3d

if __name__ == "__main__":
    xyz_path = "/Users/lihongbo/Desktop/code/Gauss_recon_py/data/01_82-block.xyz"
    point_cloud = utils.read_xyz_file(xyz_path)
    sampled_pc = utils.farthest_point_sampling(point_cloud, 10000)
    normals = utils.estimate_normals(sampled_pc, 0.1)
    # Visualize the point cloud
    vis.ps_vis_vector_field(sampled_pc, normals)
