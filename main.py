from tool import utils
import numpy as np
import torch
from gaussian.particle import Particle
from gaussian.pc_aux import PC_aux
from gaussian.loss import Loss_Func
from rvd.RVD_cpp import run_rvd
if __name__ == "__main__":
    xyz_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/data/01_82-block.xyz"
    site_num = 10000
    epoch = 300
    torch.cuda.set_device(2)
    torch.manual_seed(42)

    point_cloud = utils.read_xyz_file(xyz_path)

    pc_aux = PC_aux(point_cloud)

    particles = Particle(pc_aux, site_num)

    optimizer = torch.optim.AdamW([particles.site_points], lr=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5)

    loss_func = Loss_Func()
    for i in range(epoch):
        optimizer.zero_grad()
        loss = loss_func.cal_loss(particles)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"epoch: {i}, loss: {loss.item()}")

        particles.constrain_sites()

    result_points = particles.site_points.detach().cpu().numpy()

    xyz_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/result.xyz"
    np.savetxt(xyz_path,
               result_points.squeeze(0), fmt="%.6f")

    geo_path = "/home/hongbo/Desktop/code/Gaussian_recon/cmake-build-debug/bin/surface_reconstruction"
    output_path = "/home/hongbo/Desktop/code/Gaussian_recon_py/results/result.obj"
    run_rvd(geo_path, xyz_path, output_path)
