'''
Camera synchronization functions
Code from https://github.com/facebookresearch/SyncMatch/blob/main/syncmatch/models/synchronization.py
'''
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
convention: Rt_{w2c} @ World[3,n] = Camera[3,n]
Pointclouds are 3xN (with invisible 4-th row of 1s
Left-multiplication maps world to camera: P @ X_world = X_cam, P is 4x4, P is extrinsics of the camera
P_ij = P_j @ P_i^{-1}.
P_ij @ X_i = X_j
P = [[ R t ]
     [ 0 1 ]]
"""
import torch


def make_Rt(R, t):
    """
    Encode the transformation X -> X @ R + t where X has shape [n,3]
    """
    Rt = torch.cat([R.transpose(-2, -1), t[..., None]], dim=-1)
    pad = torch.zeros_like(Rt[..., 2:3, :])
    pad[..., -1] = 1.0
    Rt = torch.cat((Rt, pad), dim=-2)
    return Rt


def split_Rt(Rt):
    """
    Split SE(3) into SO(3) and R^3
    """
    return Rt[..., :3, :3], Rt[..., :3, 3]

def SE3_inverse(P):
    R_inv = P[..., :3, :3].transpose(-2, -1)
    t_inv = -1 * R_inv @ P[..., :3, 3:4]
    bottom_row = P[..., 3:4, :]
    Rt_inv = torch.cat((R_inv, t_inv), dim=-1)
    P_inv = torch.cat((Rt_inv, bottom_row), dim=-2)
    return P_inv


def camera_chaining(Ps, confidence, N):
    """Synchronizes cameras by chaining adjacent views:
        P_{0, 3} = P_{2, 3} @ P_{1, 2} @ P_{0, 1}
    Args:
        Ps (dict): Pairwise view estimates Ps[(i, j)] is transform i -> j
        confidence (dict): confidence for pairwise estimates, not used for chaining.
        N (int): number of views
    Returns:
        FloatTensor: synchronzed pairwise transforms (batch, 4N, 4N)
    """
    for i in range(N - 1):
        j = i + 1
        assert (i, j) in Ps

    # (i,j) are left over from the loop above.
    batch, _, _ = Ps[(i, j)].shape
    device = Ps[(i, j)].device

    L = [torch.eye(4, device=device)[None].expand(batch, 4, 4)]
    for i in range(N - 1):
        j = i + 1
        L.append(Ps[(i, j)] @ L[-1])

    L = torch.stack(L, 1)

    return L


def camera_synchronization(
    Ps,
    confidence,
    N,
    squares=10,
    so3_projection=True,
    normalize_confidences=True,
    double=True,
    center_first_camera=False,
):
    """Applies the proposed synchronization algorithm where the pairwise matrix
    is formed and iterative matrix multiplication is applied for synchronization.
    Args:
        Ps (dict): Ps[(i, j)] is pairwise estimate for i -> j
        confidence (dict): conf[(i, j)] is confidence in pairwise estimates
        N (int): number of views
        squares (int, optional): number of matrix multipliactions. Defaults to 10.
        so3_projection (bool, optional): reproject onto SO(3) during optimization
        normalize_confidences (bool, optional): normalize conf colum to 1
        double (bool, optional): run optimization in float64; good for stability
        center_first_camera (bool, optional): return cameras around 0 or N/2 view
    Returns:
        FloatTensor: synchronzed pairwise transforms (batch, 4N, 4N)
    """
    # for 2 views, there's only 1 pairwise estimate ... no sync is possible
    if N == 2:
        return camera_chaining(Ps, confidence, N)

    _views_all = []
    for i, j in Ps:
        # sanity checks
        assert (i, j) in confidence
        assert i != j
        assert (j, i) not in Ps
        _views_all.append(i)
        _views_all.append(j)

    for vi in range(N):
        assert vi in _views_all, f"View {vi} is not in any pairwise views"

    # (i,j) are left over from the loop above.
    batch, _, _ = Ps[(i, j)].shape
    device = Ps[(i, j)].device

    # form conf_matrix; turn it into a 'stochastic' matrix
    no_entry_conf = torch.zeros(batch, device=device)
    conf = [[no_entry_conf for _ in range(N)] for _ in range(N)]

    for i, j in Ps:
        c = confidence[(i, j)]
        conf[i][j] = c
        conf[j][i] = c
        if normalize_confidences:
            conf[i][i] = conf[i][i] + c / 2
            conf[j][j] = conf[j][j] + c / 2

    if not normalize_confidences:
        for i in range(N):
            conf[i][i] = torch.ones_like(no_entry_conf)

    conf = torch.stack([torch.stack(conf_row, dim=1) for conf_row in conf], dim=1)
    if normalize_confidences:
        conf = conf / conf.sum(dim=1, keepdim=True).clamp(min=1e-9)

    # === Form L matrix ===
    no_entry_P = torch.zeros(batch, 4, 4, device=device)
    diag_entry_P = torch.eye(4, device=device)[None].expand(batch, 4, 4)
    L = [[no_entry_P for i in range(N)] for j in range(N)]

    for i in range(N):
        L[i][i] = conf[:, i, i, None, None] * diag_entry_P

    for i, j in Ps:
        c_ij = conf[:, i, j, None, None]
        c_ji = conf[:, j, i, None, None]
        L[i][j] = c_ij * SE3_inverse(Ps[(i, j)])
        L[j][i] = c_ji * Ps[(i, j)]

    L = torch.cat([torch.cat(L_row, dim=2) for L_row in L], dim=1)

    if double:  # turn into double to make it more stable
        L = L.double()

    # Raise L to the power of 2**squares
    for _ in range(squares):
        L = L @ L

    L = L.view(batch, N, 4, N, 4)

    if center_first_camera:
        L = L[:, :, :, 0, :]
    else:
        L = L[:, :, :, N // 2, :]

    mass = L[:, :, 3:, 3:]
    #print(torch.sum(mass<=0.0))
    # If mass.min() ==0, either the parameter squares neeeds to be larger, or
    # the set of edges (entries in Ps) does not span the set of cameras.
    assert mass.min().item() > 0, "2**squares, or the set of edges, is too small"
    L = L / mass.clamp(min=1e-9)

    if so3_projection:
        R_pre = L[:, :, :3, :3]

        U, _, V = torch.svd(R_pre)
        V_t = V.transpose(-1, -2)
        S = torch.det(U @ V_t)
        S = torch.cat(
            [torch.ones(*S.shape, 1, 2, device=device), S[..., None, None]], -1
        )
        R = (U * S.double()) @ V_t
        L = torch.cat([torch.cat([R, L[:, :, :3, 3:]], 3), L[:, :, 3:]], 2)

    L = L.float()

    return L


def camera_synchronization_eig(Ps, confidence, N):
    """Applies the extrinsics synchronization algorithm.
    Based on algorithm in App B2 in Gojcic et al. (CVPR 2020) with some modifications.
    Args:
        Ps (dict): Ps[(i, j)] is transformation i -> j
        confidence (dict): confidence[(i, j)] is pairwise confidence
        N (int): number of views
    Returns:
        FloatTensor: synchronzed pairwise transforms (batch, 4N, 4N)
        _type_: _description_
    """
    for i, j in Ps:
        assert (i, j) in confidence
        assert i != j
        assert (j, i) not in Ps

    # (i,j) were are left over from the loop above.
    batch, Ps_height, _ = Ps[(i, j)].shape
    device = Ps[(i, j)].device

    # === Form Conf Matrix ===
    no_entry_conf = torch.zeros(batch, device=device)
    conf = [[no_entry_conf for _ in range(N)] for _ in range(N)]

    for i, j in Ps:
        c = confidence[(i, j)]
        conf[i][j] = c
        conf[j][i] = c

    conf = torch.stack([torch.stack(conf_row, dim=1) for conf_row in conf], dim=1)

    # === Form L matrix ===
    no_entry_R = torch.zeros(batch, 3, 3, device=device)
    conf_eye = torch.eye(3)[None, :].float().to(device=device)
    L = [[no_entry_R for i in range(N)] for j in range(N)]
    B = [[] for j in range(N)]

    # add identities
    for i in range(N):
        L[i][i] = conf[:, i].sum(dim=1, keepdim=True)[:, :, None] * conf_eye

    # add off diagonal
    for i, j in Ps:
        R_ij = Ps[(i, j)][:, :3, :3]
        t_ij = Ps[(i, j)][:, :3, 3:4]
        c_ij = conf[:, i, j, None, None]

        # ij
        L[i][j] = -1 * c_ij * R_ij.transpose(-2, -1)
        B[i].append(-1 * c_ij * (R_ij.transpose(-2, -1) @ t_ij))

        # ji
        L[j][i] = -1 * c_ij * R_ij
        B[j].append(c_ij * t_ij)

    # aggregate it all
    L = torch.cat([torch.cat(L_row, dim=2) for L_row in L], dim=1).contiguous()
    B = torch.cat([sum(b) for b in B], dim=1).contiguous().squeeze(dim=2)

    # turn into double to make it more stable
    L = L.double()
    B = B.double()

    # === Get Rotations ===
    # get smallest 3 eigenvectors (first 3 columns)
    L_eval, L_evec = torch.linalg.eigh(L)
    L_evec = L_evec[:, :, :3]
    L_evec = L_evec.view(batch, N, 3, 3)
    R_det = torch.det(L_evec)
    L_evec = L_evec * R_det.mean(1).sign()[:, None, None, None]

    # apply SVD
    U, _, V = torch.svd(L_evec)
    V_t = V.transpose(-1, -2)
    R = U @ V_t

    # solve for t -- lstsq doesn't work due to missing backward
    # t = torch.linalg.lstsq(L_pad, B_pad).solution
    t = torch.linalg.pinv(L) @ B[:, :, None]
    t = t.view(batch, N, 3)

    # Form output P
    P = make_Rt(R.transpose(-2, -1), t).float()

    return P
