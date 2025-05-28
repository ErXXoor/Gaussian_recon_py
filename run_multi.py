import numpy as np
import os
import subprocess


def parse_id_list(file_path):
    id_list = []
    with open(file_path, 'r') as f:
        for line in f:
            obj_id = line.strip()
            id_list.append(obj_id)
    return id_list


if __name__ == "__main__":
    id_path = "/home/hongbo/Desktop/code/PTV3_Embedding/obj_lists/new_80.txt"
    id_list = parse_id_list(id_path)

    root_path = "/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/Thing10K_point/surface_sample_20k"

    out_root = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/80s_20k_post"

    grc_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/main.py"
    params = [
        "--input",
        "",
        "--site_num",
        "",
        "--dim",
        "8",
        "--out_xyz",
        "",
        "--out_tri",
        "",
        "--out_3d",
        "",
        "--post_process",
        "true",
    ]

    for obj_id in id_list:
        xyz_path = os.path.join(root_path, obj_id,
                                f"{obj_id}_emb.xyz")
        out_xyz = os.path.join(
            out_root, f"{obj_id}_emb_grc.xyz")
        out_tri = os.path.join(
            out_root, f"{obj_id}_emb_tri.obj")
        out_3d = out_xyz.replace(".xyz", "_3d.xyz")
        site_num = 8000

        params[1] = xyz_path
        params[3] = str(site_num)
        params[7] = out_xyz
        params[9] = out_tri
        params[11] = out_3d
        print(params)

        try:
            return_code = subprocess.call(
                ["conda", "run", "-n", "grecon", "python", grc_path] + params)
            print("Return code:", return_code)
        except Exception as e:
            print("An error occurred:", str(e))
