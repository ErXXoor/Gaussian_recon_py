from tool import utils
import numpy as np
import torch
from gaussian.particle import Particle
from gaussian.pc_aux import PC_aux
from gaussian.loss import Loss_Func
from rvd.RVD_cpp import run_rvd
from gradop.pcgrad import PCGrad
if __name__ == "__main__":
    xyz_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/data/01_82-block.xyz"
    site_num = 10000
    epoch = 400
    torch.cuda.set_device(2)
    torch.manual_seed(42)

    geo_path = "/home/hongbo/Desktop/code/Gaussian_recon/cmake-build-debug/bin/surface_reconstruction"

    point_cloud, normals = utils.read_xyz_file(xyz_path)

    pc_aux = PC_aux(point_cloud, normals)

    particles = Particle(pc_aux, site_num)

    optimizer = torch.optim.AdamW([particles.site_points], lr=1e-2)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5)

    loss_func = Loss_Func()
    for i in range(epoch):
        loss = loss_func.cal_loss(particles)

        optimizer.zero_grad()
        loss_un = torch.stack(loss).sum()
        # loss_un = loss[1].sum()
        loss_un.backward()

        optimizer.step()
        scheduler.step()

        print(f"epoch: {i}, loss: {loss_un.item()}")

        # print(
        #     f"epoch: {i}, loss: {loss[0].sum().item()}, {loss[1].sum().item()}")

        # particles.constrain_sites()
        particles.update_normals()

        if i % 50 == 0:
            result_points = particles.site_points.detach().cpu().numpy()

            xyz_path = f"/home/hongbo/Desktop/code/Gaussian_recon_py/results/epoch_{i}.xyz"
            np.savetxt(xyz_path,
                       result_points.squeeze(0), fmt="%.6f")

            output_path = f"/home/hongbo/Desktop/code/Gaussian_recon_py/results/epoch_{i}.obj"
            run_rvd(geo_path, xyz_path, output_path)

    result_points = particles.site_points.detach().cpu().numpy()

    xyz_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/result.xyz"
    np.savetxt(xyz_path,
               result_points.squeeze(0), fmt="%.6f")

    output_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/result.obj"
    run_rvd(geo_path, xyz_path, output_path)
