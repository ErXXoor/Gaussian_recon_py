from tool import utils
import numpy as np
import torch
from gaussian.particle import Particle
from gaussian.pc_aux import PC_aux
from gaussian.loss import Loss_Func
from rvd.RVD_cpp import run_rvd, run_rvd_hd
from rvd.RVD import rvd_rec
from gradop.pcgrad import PCGrad
import os
import argparse


def gaussian_recon(mesh_path, site_num, dim, out_path, verbose=False):
    epoch = 200
    torch.cuda.set_device(0)
    torch.manual_seed(42)

    geo_path = "/home/hongbo/Desktop/code/SimplexCVT_recon/cmake-build-debug/src/src"

    point_cloud, normals = utils.read_xyz_file(mesh_path)

    pc_aux = PC_aux(point_cloud, normals, dim)

    particles = Particle(pc_aux, site_num, dim)

    optimizer = torch.optim.AdamW([particles.optimize_site_points], lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.6)

    loss_func = Loss_Func()
    for i in range(epoch):

        # particles.update_sigma()

        loss = loss_func.cal_loss(particles, epoch=i)

        optimizer.zero_grad()
        loss_un = torch.stack(loss).sum()
        # loss_un = loss[0].sum()
        loss_un.backward()

        optimizer.step()
        scheduler.step()

        print(f"epoch: {i}, loss: {loss_un.item()}")

        # print(
        #     f"epoch: {i}, loss: {loss[0].sum().item()}, {loss[1].sum().item()}")

        particles.update_site_points()
        # particles.update_normals()

        if dim == 3:
            particles.constrain_sites()
        else:
            particles.constrain_sites_hd()

        # particles.update_normals()

        if verbose and i % 50 == 0:
            result_points = particles.optimize_site_points.detach(
            ).cpu().numpy()

            xyz_path = f"{out_path}/epoch_{i}.xyz"
            np.savetxt(xyz_path,
                       result_points.squeeze(0), fmt="%.6f")

            output_path = f"{out_path}/epoch_{i}.obj"
            # run_rvd(geo_path, xyz_path, output_path)
            run_rvd_hd(geo_path, dim, xyz_path, output_path)

    if dim == 3:
        particles.constrain_sites()
    else:
        particles.constrain_sites_hd()

    result_points = particles.optimize_site_points.detach(
    ).cpu().numpy()

    xyz_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/result.xyz"
    np.savetxt(xyz_path,
               result_points.squeeze(0), fmt="%.6f")

    output_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/result.obj"
    # run_rvd(geo_path, xyz_path, output_path)
    run_rvd_hd(geo_path, dim, xyz_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/home/hongbo/Desktop/code/PTV3_Embedding/outputs/ptv3_01/eval/59941_norm_eval_emb.xyz",
                        help='input point cloud file')
    parser.add_argument('--site_num', type=int, default=8000,
                        help='number of site points')
    parser.add_argument('--dim', type=int, default=8,
                        help='dimension of the point cloud')
    parser.add_argument('--out_path', type=str, default='/home/hongbo/Desktop/code/Gaussian_recon_py/results/',
                        help='output path for the results')
    parser.add_argument('--verbose', action='store_true',
                        help='print verbose output')

    args = parser.parse_args()

    input_path = args.input
    out_path = args.out_path
    site_num = args.site_num
    dim = args.dim
    gaussian_recon(input_path, site_num, dim, out_path, True)
