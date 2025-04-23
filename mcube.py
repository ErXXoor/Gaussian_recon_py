import open3d as o3d
import numpy as np
from glob import glob
import os

if __name__ == "__main__":
    folder_path = "/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/dora-bench-256/mnt/bn/dora-bench-1/dora_bench/sub_temp_samples/xyz_points"

    output_folder = "/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/dora-bench-256/mnt/bn/dora-bench-1/dora_bench/sub_temp_samples/xyz_points_normalized"

    path_list = glob(os.path.join(folder_path, "*.xyz"))

    print(path_list[0])

    pcd_data = np.loadtxt(path_list[0])
    points = pcd_data[:, :3]
    normals = pcd_data[:, 3:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.normals = o3d.utility.Vector3dVector(normals)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=12,
    )

    densities = np.asarray(density)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.vertex_colors = o3d.utility.Vector3dVector()

    out_path = f"/home/hongbo/Desktop/code/Gaussian_recon_py/temp/{os.path.basename(path_list[0]).split('.')[0]}.obj"
    print(out_path)
    o3d.io.write_triangle_mesh(out_path, mesh)
