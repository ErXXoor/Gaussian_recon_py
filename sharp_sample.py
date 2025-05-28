import bpy
import bmesh
import math
import numpy as np


if __name__ == "__main__":
    mesh_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/data/72879_sf_norm.obj"
    angle_threshold = 15
    sharpness_threshold = math.radians(angle_threshold)

    bpy.ops.wm.obj_import(filepath=mesh_path)
    # bpy.ops.wm.stl_import(filepath=mesh_path)
    # bpy.ops.wm.ply_import(filepath=mesh_path)
    # 假设导入的对象是当前活动对象
    obj = bpy.context.selected_objects[0]

    # 进入Edit模式
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # 确保在边模式下选择
    bpy.ops.mesh.select_mode(type="EDGE")

    # 选择Sharp Edge
    bpy.ops.mesh.edges_select_sharp(sharpness=sharpness_threshold)

    # 打印Sharp Edge
    bpy.ops.object.mode_set(mode='OBJECT')  # 临时切换回Object模式以访问选择状态
    # mesh = obj.data

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    sharp_edges = [edge for edge in bm.edges if edge.select]

    sharp_edges_vertices = []
    link_normal1 = []
    link_normal2 = []
    sharp_edges_angle = []
    # 不重复点集
    vertices_set = set()
    for edge in sharp_edges:
        vertices_set.update(edge.verts[:])  # 不重复点集

        # 收集 sharp edges 的顶点对 index
        sharp_edges_vertices.append([edge.verts[0].index, edge.verts[1].index])

        normal1 = edge.link_faces[0].normal
        normal2 = edge.link_faces[1].normal

        link_normal1.append(normal1)
        link_normal2.append(normal2)

        if normal1.length == 0.0 or normal2.length == 0.0:
            sharp_edges_angle.append(0.0)
        # Compute the angle between the two normals
        else:
            sharp_edges_angle.append(math.degrees(normal1.angle(normal2)))

    vertices = []
    for vertex in vertices_set:
        vertices.append(vertex.co)
    normal_arr1 = np.array(link_normal1)
    normal_arr2 = np.array(link_normal2)
    aaa = 0
