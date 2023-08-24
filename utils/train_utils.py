import torch
import torch.nn as nn
import numpy as np
import dataclasses
import collections
import os
import random
from pytorch3d.renderer.cameras import PerspectiveCameras

from utils.geo_utils import euler2mat


def set_camera_intrinsics(B, config, cameras=None):
    '''
    focal: focal regressed from encoder, in [B,1]
    cameras: dict of camera parameters
    return: add camera principle point at the center of image, in [B,2]
    '''
    if cameras is None:
        cameras = {}
    principal_point = config.dataset.img_size / 2
    focal = config.render.camera_focal
    p = torch.tensor([principal_point]).unsqueeze(0).repeat(B,2)
    f = torch.tensor([focal]).repeat(B)
    cameras['f'] = f
    cameras['p'] = p
    return cameras


def set_camera_pose_canonical(B, trans_z):
    R = torch.tensor([[1.0,0,0], [0,1,0], [0,0,1]])  # [3,3]
    t = torch.tensor([0.0, 0.0, trans_z]).view(3,1)
    RT = torch.cat((R,t), dim=-1)  # [3, 4]
    return RT.unsqueeze(0).repeat(B, 1, 1)


def change_camera_pose(cameras, new_pose):
    B = cameras['f'].shape[0]

    R = new_pose[:, :3, :3]
    T = new_pose[:, :3, 3].view(-1,3)

    f = cameras['f']
    p = cameras['p']  # [B,2]
    K = torch.eye(3).unsqueeze(0).repeat(B,1,1)  # [B,3,3]
    K[:,0,0] = f.clone()
    K[:,1,1] = f.clone()
    K[:,0,2] = p[:,0]
    K[:,1,2] = p[:,1]
    cameras ={
        'R': R,
        'T': T,
        'f': f,
        'p': p,
        'K': K
    }
    return cameras


def choose_camera_pose_NVS(B, all_pose):
    idx = random.choices(range(all_pose.shape[0]), k=B)
    chosen_pose = all_pose[idx]  # [B,3,4]
    return chosen_pose


def rotate_camera_pose(B, t, canonical_pose, rotMat, cameras):
    '''
    B: batch size
    t: frame number
    canonical_pose: canoinical camera pose, in [B,3,4]
    rotMat: camera relative pose from caonical frame to each frame, in [B*(t-1),3,4]
    '''
    device = rotMat.device

    canonical_pose_expand = canonical_pose.unsqueeze(1).repeat(1,t-1,1,1)  # [B,t-1,3,4]
    canonical_pose_expand = canonical_pose_expand.reshape(B*(t-1),3,4).to(device)   # [B*(t-1),3,4]
    
    tmp = torch.zeros(B*(t-1),1,4)
    tmp[:,:,3] = 1
    tmp = tmp.to(device)

    canonical_pose_expand = torch.cat([canonical_pose_expand, tmp], dim=1)  # [B*(t-1), 4, 4]
    rotMat_expand = torch.cat([rotMat, tmp], dim=1)  # [B*(t-1), 4, 4]

    camera_pose = torch.matmul(canonical_pose_expand, rotMat_expand)  # [B*(t-1),4,4]
    camera_pose = camera_pose[:,:3,:].reshape(B,t-1,3,4)
    camera_pose = torch.cat([canonical_pose.unsqueeze(1).to(device), camera_pose], dim=1)  # [B,t,3,4]
    camera_pose = camera_pose.reshape(B*t,3,4)

    f = cameras['f']  # [B*t]
    p = cameras['p']  # [B*t,2]
    K = torch.eye(3).unsqueeze(0).repeat(B*t,1,1)  # [B*t,3,3]
    K[:,0,0] = f.clone()
    K[:,1,1] = f.clone()
    K[:,0,2] = p[:,0]
    K[:,1,2] = p[:,1]
    cameras ={
        'R': camera_pose[:, :3, :3].cpu(),  # [B*t,3,3]
        'T': camera_pose[:, :3, 3].view(-1,3).cpu(),  # [B*t,3]
        'RT': camera_pose.cpu(), # [B*t, 3, 4]
        'f': f,
        'p': p,
        'K': K
    }
    return cameras


def rotMat_mul(rot_base, rot_change):
    '''
    rot_base, rot_change: rotation matrix in [B,3,4]
    '''    
    device = rot_base.device
    B = rot_base.shape[0]

    tmp = torch.zeros(B,1,4)
    tmp[:,:,3] = 1
    tmp = tmp.to(device)

    rot_base_expend = torch.cat([rot_base, tmp], dim=1)  # [B,4,4]
    rot_change_expend = torch.cat([rot_change, tmp], dim=1)  # [B,4,4]
    rot_res = torch.matmul(rot_base_expend, rot_change_expend)  # [B,4,4]
    return rot_res[:,:3,:]
    

def dataclass_to_cuda_(obj):
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj


def try_to_cuda(t):
    try:
        t = t.cuda()
    except AttributeError:
        pass
    return t


def dict_to_cuda(batch):
    return {k: try_to_cuda(v) for k, v in batch.items()}


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def adjust_lr(config, optimizer, iter_num, adjust_iter_num):
    if config.dataset.name == 'omniobject3d':
        warmup_iter = 500
        max_lr = config.train.lr
        lr = max_lr * iter_num / warmup_iter
    if iter_num == adjust_iter_num[0]:
        lr = config.train.lr * 0.5
    elif iter_num == adjust_iter_num[1]:
        lr = config.train.lr * 0.25
    elif iter_num == adjust_iter_num[2]:
        lr = config.train.lr * 0.125
    elif iter_num == adjust_iter_num[3]:
        lr = config.train.lr * 0.0625
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)


def compute_confidece_gt(pred, gt):
    '''
    https://github.com/kentsommer/tensorflow-posenet/blob/master/test.py
    pred, gt in shape [b,7]
    '''
    device = pred.device
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    B = pred.shape[0]

    confs = []
    for i in range(B):
        q_pred = pred[i, :4]
        q = gt[i, :4]
        d = np.abs(np.sum(np.multiply(q_pred, q)))
        theta = 2 * np.arccos(d) * 180 / np.pi
        conf_pose = (np.cos(theta) + 1) / 2

        t_pred = pred[i, 4:]
        t = gt[i, 4:]
        t_error = np.linalg.norm(t_pred - t)
        if t_error > 1.0:
            t_error = 1.0
        conf_trans = 1 - t_error
        
        conf = (conf_pose + conf_trans) / 2
        confs.append(conf)
    
    return torch.tensor(confs).float().to(device)


def fix_bn_affine_params(model):
    for name, child in (model.named_children()):
        if isinstance(child, nn.BatchNorm1d) or isinstance(child, nn.BatchNorm2d) or isinstance(child, nn.BatchNorm3d):
            if hasattr(child, 'weight'):
                child.weight.requires_grad_(False)
            if hasattr(child, 'bias'):
                child.bias.requires_grad_(False)
                

def fix_bn_running_stat(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
        if isinstance(module, nn.BatchNorm3d):
            module.eval()
             
