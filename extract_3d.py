import numpy as np


def extract_3d_points(particle_hd: np.array, output_file: str):
    particle_3d = particle_hd[:, :3]
    np.savetxt(output_file, particle_3d)


if __name__ == "__main__":
    particle_path = "/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/Thing10K_point/surface_sample_20k/72203/72203_emb.xyz"
    output_file = "/home/hongbo/Desktop/code/Gaussian_recon_py/temp/72203_emb_3d.xyz"
    particle_hd = np.loadtxt(particle_path, dtype=np.float32)
    extract_3d_points(particle_hd, output_file)
    print(f"Extracted 3D points to {output_file}")
