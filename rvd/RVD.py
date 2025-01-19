import torch
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.io import save_obj
EPS = 5e-6


def fix_precision(tensor, precision=EPS):
    # change to abs(tensor)<precision
    tensor[(tensor > 0) & (tensor < precision)] = 0
    tensor[(tensor > -precision) & (tensor < 0)] = 0
    return tensor


def generate_couple_list(num):
    couple_list = []
    for i in range(num):
        for j in range(i + 1, num):
            couple_list.append((i, j))
    couple_list = torch.tensor(couple_list, dtype=torch.long).cuda()
    return couple_list


def compute_intersection(coef_mat):
    # solve the equation if rank is 3
    rank_A = torch.linalg.matrix_rank(coef_mat[:, :, :3])

    coef_idx = torch.nonzero(rank_A == 3)

    sol = torch.linalg.solve(
        coef_mat[coef_idx, :, :3], -coef_mat[coef_idx, :, 3])

    return sol, coef_idx


def clip_plane(ct_point, itsct_list, bsct_list):
    sign_ct = torch.sign(torch.matmul(
        ct_point, bsct_list[:, :3].t()) + bsct_list[:, 3])
    itsct_test = torch.matmul(
        itsct_list, bsct_list[:, :3].t()) + bsct_list[:, 3]
    itsct_test = fix_precision(itsct_test)
    sign_itsct = torch.sign(itsct_test)

    side_decision = sign_ct * sign_itsct
    side_decision = torch.where(side_decision < 0, torch.zeros_like(
        side_decision), torch.ones_like(side_decision))

    side_decision = side_decision.all(dim=-1)
    final_itsct_idx = torch.where(side_decision)[0]
    return final_itsct_idx


def plane_intersection(plane_neighbor, plane_center, couple_list):
    plane_coef_couples = plane_neighbor[couple_list, :]

    plane_center_expand = plane_center.unsqueeze(
        0).expand(couple_list.shape[0], -1).unsqueeze(-2)
    plane_coef_couples = torch.cat(
        (plane_coef_couples, plane_center_expand), dim=-2)

    itsct_point, couple_idx = compute_intersection(plane_coef_couples)

    itsct_couples = couple_list[couple_idx]
    return itsct_point.to(torch.float32), itsct_couples


def rvd_rec(input_points: torch.tensor, point_normals: torch.tensor, K=20):
    knn_results = knn_points(input_points, input_points, K=K)
    point_ngbrs = knn_gather(input_points, knn_results.idx[..., 1:])

    point_ngbrs = point_ngbrs.squeeze(0)
    point_cts = input_points.unsqueeze(-2).squeeze(0).expand_as(point_ngbrs)
    dir_norm = torch.norm(point_ngbrs-point_cts, dim=-1, keepdim=True)
    dir_vec = (point_ngbrs-point_cts) / dir_norm

    bisct = (point_ngbrs + point_cts) / 2
    bisct_plane = torch.cat(
        (dir_vec, -torch.sum(bisct*dir_vec, dim=-1, keepdim=True)), dim=-1)

    col_w_tgt = -torch.sum(input_points * point_normals, dim=-1)
    tangent_plane = torch.cat(
        (point_normals, col_w_tgt.unsqueeze(-1)), dim=-1).squeeze(0)

    couple_list = generate_couple_list(K-1)

    face_list = []
    itsc_list = []
    vert_list = []

    for i in range(tangent_plane.shape[0]):
        itsc, itsct_couples = plane_intersection(
            bisct_plane[i], tangent_plane[i], couple_list)

        final_itsct_idx = clip_plane(point_cts[i, 0, :],
                                     itsc,
                                     bisct_plane[i])
        final_itsc = itsc[final_itsct_idx]
        itsc_list.append(final_itsc)

        ngbr_idx = itsct_couples[final_itsct_idx].squeeze(-2)

        ngbr_idx_global = knn_results.idx[0, i, 1:][ngbr_idx]
        ct_idx_global = torch.tensor(
            i).repeat(ngbr_idx_global.shape[0]).unsqueeze(-1).cuda()
        faces = torch.cat((ct_idx_global, ngbr_idx_global), dim=-1)
        face_list.append(faces)

    vert_list = input_points.squeeze(0)
    face_list = torch.cat(face_list, dim=0)
    save_obj("/home/hongbo/Desktop/code/Gaussian_recon_py/results/rvd.obj",
             vert_list, face_list)
