import robust_laplacian
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

if __name__ == "__main__":
    pcd_path = "/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/Thing10K_point/surface_sample_20k/70558/70558_norm.xyz"

    out_filename = "/home/hongbo/Desktop/code/Gaussian_recon_py/temp/70558_ngbr20.obj"

    pcd = np.loadtxt(pcd_path, dtype=np.float32)
    pcd_3d = pcd[:, :3]
    # L, _ = robust_laplacian.point_cloud_laplacian(pcd_3d,
    #                                               n_neighbors=30)

    # eigenvals, eigenvecs = eigsh(
    #     L, k=20, which='SM')

    # eigenvals = eigenvals[1:]
    # eigenvecs = eigenvecs[:, 1:]

    # diffusion_coords = eigenvecs * np.exp(-eigenvals * 1.0)
    # nbrs = NearestNeighbors(
    #     n_neighbors=12).fit(diffusion_coords)
    # dists, indices = nbrs.kneighbors(diffusion_coords)

    nbrs = NearestNeighbors(n_neighbors=20).fit(pcd_3d)
    dists, indices = nbrs.kneighbors(pcd_3d)
    # indices = indices.astype(np.int32)[:, 5:]

    # Convert indices to edges
    edges = []
    for i in range(len(indices)):
        for j in range(1, len(indices[i])):
            edges.append((i, indices[i][j]))
    edges = np.array(edges)

    # Write .obj file
    with open(out_filename, "w") as f:
        # Write vertices
        for p in pcd_3d:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

        # Write edges (1-based indexing)
        for i, j in edges:
            f.write(f"l {i+1} {j+1}\n")
