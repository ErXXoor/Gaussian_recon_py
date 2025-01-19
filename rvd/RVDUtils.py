import torch
from pytorch3d.ops import knn_points
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
    coef_idx = torch.where(rank_A == 3)[0]

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

    plane_center_expand = plane_center.repeat(
        couple_list.shape[0], 1).unsqueeze(-2)
    plane_coef_couples = torch.cat(
        (plane_coef_couples, plane_center_expand), dim=-2)

    itsct_point, couple_idx = compute_intersection(plane_coef_couples)

    itsct_couples = couple_list[couple_idx]
    return itsct_point.to(torch.float32), itsct_couples


def weight_to_edge(input_points, point_weights):
    points_ori = input_points.view(
        point_weights.shape[0], point_weights.shape[1], -1)
    points_nbr = points_ori[:, 1:, :3]
    points_ct = points_ori[:, 0, :3].unsqueeze(
        1).repeat(1, points_nbr.shape[1], 1)
    nbr_ct = torch.square(points_nbr - points_ct)
    nbr_ct = torch.sum(nbr_ct, dim=-1)

    w_c = point_weights[:, 0].unsqueeze(-1).repeat(1, points_nbr.shape[1])
    w_nbr = point_weights[:, 1:]
    batch_weight = (w_c - w_nbr + nbr_ct) / (2 * nbr_ct)
    return batch_weight


def weight_rvd_rec(input_point, edge_weights, point_normals):
    points_ori = input_point.view(
        edge_weights.shape[0], edge_weights.shape[1] + 1, -1)

    points_nbr = points_ori[:, 1:, :3]
    points_ct = points_ori[:, 0, :3].unsqueeze(
        1).repeat(1, points_nbr.shape[1], 1)

    neighbor_num = points_nbr.shape[1]

    dir_norm = torch.linalg.norm(points_nbr - points_ct, dim=-1)
    dir_vec = (points_nbr - points_ct) / dir_norm.unsqueeze(-1)

    edge_weights = edge_weights.unsqueeze(-1)
    w_bisct = torch.mul((edge_weights), points_ct) + \
        torch.mul(1 - edge_weights, points_nbr)

    col_w = -1 * torch.sum(w_bisct * dir_vec, dim=-1)
    bisct_plane = torch.cat((dir_vec, col_w.unsqueeze(-1)), dim=-1)

    col_w_tgt = -1 * torch.sum(
        points_ct[:, 0, :] * point_normals,
        dim=-1)
    tangent_plane = torch.cat((point_normals, col_w_tgt.unsqueeze(-1)), dim=-1)

    couple_list = generate_couple_list(neighbor_num)

    batch_vert_list = []
    batch_face_list = []
    batch_itsc_list = []
    batch_vertid_list = []
    for i in range(tangent_plane.shape[0]):
        itsc, itsct_couples = plane_intersection(
            bisct_plane[i], tangent_plane[i], couple_list)

        final_itsct_idx = clip_plane(points_ct[i, 0, :],
                                     itsc,
                                     bisct_plane[i])

        final_itsc = itsc[final_itsct_idx]
        batch_itsc_list.append(final_itsc)

        ngbr_idx = itsct_couples[final_itsct_idx]
        ngbr_idx += 1
        vert_idx_list = list(set(ngbr_idx.flatten().cpu().numpy()))
        vert_idx_list.insert(0, 0)

        vert_list = points_ori[i, vert_idx_list, :3]
        batch_vert_list.append(vert_list)

        face_list = torch.tensor([vert_idx_list.index(x)
                                  for x in ngbr_idx.flatten()], dtype=torch.long)
        face_list = face_list.reshape(ngbr_idx.shape)
        face_list = torch.cat(
            (torch.tensor([0]).repeat(face_list.shape[0],
                                      1),
             face_list), dim=-1).cuda()
        batch_face_list.append(face_list)

    return batch_itsc_list, batch_vert_list, batch_face_list


def weight_rvd_rec_eval(input_point, edge_weights, point_normals, idx_map):
    points_ori = input_point.view(
        edge_weights.shape[0], edge_weights.shape[1] + 1, -1)

    points_nbr = points_ori[:, 1:, :3]
    points_ct = points_ori[:, 0, :3].unsqueeze(
        1).repeat(1, points_nbr.shape[1], 1)

    neighbor_num = points_nbr.shape[1]

    dir_norm = torch.linalg.norm(points_nbr - points_ct, dim=-1)
    dir_vec = (points_nbr - points_ct) / dir_norm.unsqueeze(-1)

    edge_weights = edge_weights.unsqueeze(-1)
    w_bisct = torch.mul((1 - edge_weights), points_ct) + \
        torch.mul(edge_weights, points_nbr)

    col_w = -1 * torch.sum(w_bisct * dir_vec, dim=-1)
    bisct_plane = torch.cat((dir_vec, col_w.unsqueeze(-1)), dim=-1)

    col_w_tgt = -1 * torch.sum(
        points_ct[:, 0, :] * point_normals,
        dim=-1)
    tangent_plane = torch.cat((point_normals, col_w_tgt.unsqueeze(-1)), dim=-1)

    couple_list = generate_couple_list(neighbor_num)

    batch_face_list = []
    batch_itsc_list = []
    for i in range(tangent_plane.shape[0]):
        itsc, itsct_couples = plane_intersection(
            bisct_plane[i], tangent_plane[i], couple_list)

        final_itsct_idx = clip_plane(points_ct[i, 0, :],
                                     itsc,
                                     bisct_plane[i])

        final_itsc = itsc[final_itsct_idx]
        ct_pos = points_ct[i, 0, :].unsqueeze(0)
        final_itsc_ct = torch.cat((ct_pos, final_itsc), dim=0)
        batch_itsc_list.append(final_itsc_ct)

        final_itsc_couples = itsct_couples[final_itsct_idx] + 1
        global_idx = idx_map[i, final_itsc_couples]

        ct_idx = idx_map[i, 0].repeat(global_idx.shape[0], 1)
        faces = torch.cat((ct_idx, global_idx), dim=-1)
        batch_face_list.append(faces)

    return batch_itsc_list, batch_face_list
