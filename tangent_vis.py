import numpy as np
from gaussian.pc_aux import PC_aux
from tool import utils


def load_hd_mesh(mesh_path):
    hd_verts = []
    faces = []
    with open(mesh_path, 'r') as f:
        for line in f:
            if line.startswith("v "):
                values = [float(x) for x in line.split()[1:]]
                hd_verts.append(values)
            elif line.startswith("f "):
                values = [int(x) - 1 for x in line.split()[1:]]
                faces.append(values)
    return hd_verts, faces


if __name__ == "__main__":
    hd_pc_path = "/home/hongbo/Desktop/code/PTV3_Embedding/outputs/ptv3_new_02/eval/82257_eval_best_emb.xyz"

    # hd_pc_path = "/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/Thing10K_point/surface_sample_20k/65617/65617_emb.xyz"

    # hd_tri_path = "/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/Thing10k_tetwild/hd_output_sqrt/437347/437347_emb.obj"

    output_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/temp/82257_tgt_02.obj"

    point_cloud, normals = utils.read_xyz_file(hd_pc_path)

    # hd_verts, _ = load_hd_mesh(hd_tri_path)
    # point_cloud = np.concatenate([point_cloud, np.array(hd_verts)], axis=0)

    pc_aux = PC_aux(point_cloud, normals, dim=8)

    radius = 0.01
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    circle_2d = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    tg_0 = pc_aux.hd_eig0.squeeze(0).cpu().numpy()
    tg_1 = pc_aux.hd_eig1.squeeze(0).cpu().numpy()

    tg_basis = np.stack([tg_0, tg_1], axis=2)

    circle_2d = circle_2d[None, :, :]  # shape (1, k, 2)

    # Batch matrix multiply: (n, k, 2) @ (n, 2, d) → (n, k, d)
    mapped_circle = np.matmul(
        circle_2d, tg_basis.transpose(0, 2, 1))  # (n, k, d)

    mapped_circle += point_cloud[:, None, :]  # (n, k, d)

    # point_cloud = point_cloud[8000:]
    # mapped_circle = mapped_circle[8000:]

    n = point_cloud.shape[0]  # number of points in the point cloud
    k = mapped_circle.shape[1]  # number of points in the circle

    circles_size = mapped_circle.shape[0] * mapped_circle.shape[1]
    vertices = np.concatenate([point_cloud, mapped_circle.reshape(
        circles_size, 8)], axis=0)  # shape: (n + n*k, d)

    # Step 4: Generate triangle fan faces
    faces = []
    for i in range(n):
        center_idx = i + 1  # .obj is 1-based
        circle_start = n + i * k  # index in flattened circle array
        for j in range(k):
            p1 = circle_start + j + 1
            p2 = circle_start + ((j + 1) % k) + 1
            faces.append((center_idx, p1, p2))

    with open(output_path, "w") as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {' '.join(map(str, v[:3]))}\n")

        # Write faces
        for face in faces:
            f.write(f"f {' '.join(map(str, face))}\n")
