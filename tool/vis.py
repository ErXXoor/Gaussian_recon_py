import polyscope as ps
import open3d as o3d
import numpy as np
ps.init()


def o3d_vis(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pc],
                                      window_name="Point Cloud Visualization",
                                      width=800, height=600,
                                      left=50, top=50,
                                      point_show_normal=False)


def ps_vis_vector_field(points, vectors):
    ps_cloud = ps.register_point_cloud("points", points)
    ps_cloud.add_vector_quantity("vectors", vectors, enabled=True)
    ps.show()
