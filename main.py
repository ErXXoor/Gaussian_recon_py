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


def gaussian_recon(mesh_path, site_num, out_path, verbose=False):
    epoch = 200
    torch.cuda.set_device(0)
    torch.manual_seed(42)

    # geo_path = "/home/hongbo/Desktop/code/Gaussian_recon/cmake-build-debug/bin/surface_reconstruction"

    geo_path = "/home/hongbo/Desktop/code/geogram/cmake-build-release/bin/co3netest"

    point_cloud, normals = utils.read_xyz_file(mesh_path)

    pc_aux = PC_aux(point_cloud, normals)

    particles = Particle(pc_aux, site_num)

    optimizer = torch.optim.AdamW([particles.optimize_site_points], lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.6)

    loss_func = Loss_Func()
    for i in range(epoch):

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

        # particles.constrain_sites()
        particles.update_site_points()
        particles.constrain_sites_hd()

        particles.update_normals()

        if verbose and i % 50 == 0:
            result_points = particles.optimize_site_points.detach(
            ).cpu().numpy()

            xyz_path = f"{out_path}/epoch_{i}.xyz"
            np.savetxt(xyz_path,
                       result_points.squeeze(0), fmt="%.6f")

            output_path = f"{out_path}/epoch_{i}.obj"
            # run_rvd(geo_path, xyz_path, output_path)
            run_rvd_hd(geo_path, xyz_path, output_path)

    # particles.constrain_sites()
    # particles.constrain_sites_hd()

    result_points = particles.optimize_site_points.detach(
    ).cpu().numpy()

    xyz_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/result.xyz"
    np.savetxt(xyz_path,
               result_points.squeeze(0), fmt="%.6f")

    output_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/result.obj"
    # run_rvd(geo_path, xyz_path, output_path)
    run_rvd_hd(geo_path, xyz_path, output_path)


if __name__ == "__main__":
    input_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/data/think10k107910.xyz"
    out_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/"
    site_num = 10000
    gaussian_recon(input_path, site_num, out_path, True)
