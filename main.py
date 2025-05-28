from tool import utils
import numpy as np
import torch
from gaussian.particle import Particle
from gaussian.pc_aux import PC_aux
from gaussian.loss import Loss_Func
from rvd.RVD_cpp import run_rvd, run_rvd_hd
from extract_3d import extract_3d_points
from gradop.pcgrad import PCGrad
import os
import argparse


def gaussian_recon(mesh_path, site_num, dim, out_xyz, out_tri, out_3d, post_process, verbose=False, verbose_root=None):
    epoch = 200
    torch.cuda.set_device(0)
    torch.manual_seed(42)

    geo_path = "/home/hongbo/Desktop/working/SimplexCVT_recon/cmake-build-debug/src/src"

    point_cloud, normals = utils.read_xyz_file(mesh_path)

    pc_aux = PC_aux(point_cloud, normals, dim)

    particles = Particle(pc_aux, site_num, dim)

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

            xyz_path = f"{verbose_root}/epoch_{i}.xyz"
            np.savetxt(xyz_path,
                       result_points.squeeze(0), fmt="%.6f")

            output_path = f"{verbose_root}/epoch_{i}.obj"
            # run_rvd(geo_path, xyz_path, output_path)
            run_rvd_hd(geo_path, dim, xyz_path, output_path)

    if dim == 3:
        particles.constrain_sites()
    else:
        particles.constrain_sites_hd()

    result_points = particles.optimize_site_points.detach(
    ).cpu().numpy().squeeze(0)

    if out_3d is not None:
        extract_3d_points(result_points, out_3d)

    np.savetxt(out_xyz, result_points)

    run_rvd_hd(geo_path, dim, out_xyz, out_tri, post_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/Thing10K_point/surface_sample_20k/42155/42155_emb.xyz",
                        help='input point cloud file')
    parser.add_argument('--site_num', type=int, default=8000,
                        help='number of site points')
    parser.add_argument('--dim', type=int, default=8,
                        help='dimension of the point cloud')
    parser.add_argument('--out_xyz', type=str, default='/home/hongbo/Desktop/code/PTV3_Embedding/outputs/ptv3_new_03/eval/70558_eval_best_grc.xyz',
                        help='output path for the results')
    parser.add_argument('--out_tri', type=str,
                        default='/home/hongbo/Desktop/code/PTV3_Embedding/outputs/ptv3_new_03/eval/70558_eval_best_tri.obj',)
    parser.add_argument(
        '--out_3d', type=str, default='/home/hongbo/Desktop/code/PTV3_Embedding/outputs/ptv3_new_03/eval/70558_eval_best_grc_3d.xyz',)

    parser.add_argument('--post_process', type=str, default="true",
                        help='whether to post process the results')

    parser.add_argument('--verbose', action='store_true',
                        help='print verbose output')

    parser.add_argument(
        '--verbose_root', default='/home/hongbo/Desktop/code/Gaussian_recon_py/results/')

    args = parser.parse_args()

    input_path = args.input
    out_xyz = args.out_xyz
    out_tri = args.out_tri
    out_3d = args.out_3d
    post_process = args.post_process.lower() == 'true'
    site_num = args.site_num
    dim = args.dim
    verbose_root = args.verbose_root
    gaussian_recon(input_path, site_num, dim, out_xyz,
                   out_tri, out_3d, post_process, True, verbose_root)
