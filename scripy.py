import numpy as np
import fpsample

if __name__ == "__main__":
    # Load the point cloud data
    data = np.loadtxt(
        "/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/Thing10K_point/dataset_80k/37384/37384_emb.xyz")
    fps_sample_idx = fpsample.fps_sampling(data, 1000)
    print(fps_sample_idx)
