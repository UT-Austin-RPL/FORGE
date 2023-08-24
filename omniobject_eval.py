import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F

import torch.utils.data
import torch.utils.data.distributed

import argparse
from config.config import config, update_config
from utils import exp_utils, train_utils, eval_utils, vis_utils, geo_utils, sync_utils
import lpips

from models.model import FORGE as ReconModel
from dataset.kubric import Kubric
from dataset.gso import GSO
from dataset.omniobject3d import Omniobject3D
import time
from itertools import combinations
from pytorch3d.renderer import look_at_view_transform
import json
import copy

from utils.eval_utils import permute_clips
from models.model import sequence_from_distance, chose_selected

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


def run_optimization(args, config, loader, dataset, model, model_gt, lpips_vgg, output_dir, device):
    '''
    For each instance:
        1. predict initial results of each view, evaluate and choose canonical index
        2. do optimization
        3. visualize the results and do evaluation
    '''
    model.eval()
    model_gt.eval()

    model_res = model_gt if args.model_gt else model

    outfile_path = os.path.join(output_dir, 'results')
    os.makedirs(outfile_path, exist_ok=True)
    outfile_path = os.path.join(outfile_path, 'results.txt')
    posefile_path = outfile_path.replace('results.txt', 'poses_{}.pth'.format(args.exp_id))
    pose_dict = {}

    # test with batch=1
    for batch_idx, sample in enumerate(loader):
        if batch_idx % args.split_num != args.exp_id:
            continue

        seen_flag = sample['seen_flag'][0].item() > 0
        
        # initialization
        return_dict = predict_initial(model, sample, device)
        best_canonical_id, psnr, ssim, lpips, rot_error, trans_error, depth_error = evaluate_all(model, lpips_vgg, sample, dataset, return_dict, batch_idx, device, output_dir, 'before')
        pose, features = return_dict[str(best_canonical_id)]['poses_cam'], return_dict[str(best_canonical_id)]['features_raw']
        nvs_extr, gt_poses = return_dict[str(best_canonical_id)]['nvs_extr'], return_dict[str(best_canonical_id)]['gt_poses']
        gt_poses = return_dict[str(best_canonical_id)]['gt_poses']
        pose_cp = pose.clone()
        visualize_360(model_res, sample, dataset, pose, features, batch_idx, 'before', output_dir, device)

        # synchronization
        if args.sync:
            try:
                pose_sync = sync_pose(copy.deepcopy(return_dict), best_canonical_id, device)
                _, rot_error_sync, _ = refine_pose(batch_idx, model_res, sample, dataset, pose_sync.clone(), features, best_canonical_id, device, output_dir, vis=False, iter_num=1)
                if rot_error_sync < rot_error:
                    pose = pose_sync.clone()
            except:
                print('{} fail to sync poses'.format(batch_idx))
        
        # refinement
        pose_refined, rot_error_refined, trans_error_refined = refine_pose(batch_idx, model_res, sample, dataset, pose, features, best_canonical_id, device, output_dir, iter_num=args.iter_num)
        psnr_refined, ssim_refined, lpips_refined, depth_error_refined = evaluate(model_res, lpips_vgg, sample, dataset, pose_refined, features, nvs_extr, gt_poses, batch_idx, best_canonical_id, device, output_dir, 'after', eval_pose=False)
        visualize_360(model_res, sample, dataset, pose_refined, features, batch_idx, 'after', output_dir, device)

        with open(outfile_path, 'a+') as f:
            line1 = 'idx {}, seen {}, before, psnr {}, ssim {}, lpips {}, rot {}, trans {}, depth {}\n'.format(
                         batch_idx, seen_flag, psnr, ssim, lpips, rot_error, trans_error, depth_error)
            line2 = 'idx {}, seen {}, after, psnr {}, ssim {}, lpips {}, rot {}, trans {}, depth {}\n'.format(
                         batch_idx, seen_flag, psnr_refined, ssim_refined, lpips_refined, rot_error_refined, trans_error_refined, depth_error_refined)
            line = line1 + line2
            f.write(line)
        pose_dict[batch_idx] = {'before': pose_cp.detach().cpu(), 'after': pose_refined.detach().cpu(), 'gt': gt_poses}
        torch.save(pose_dict, posefile_path)

def sync_pose(return_dict, best_canonical_id, device=None):
    pose_dict, conf_dict = {}, {}
    best_canonical_pairs = []

    t = len(return_dict.keys())
    # get all pose estimation
    for it in return_dict.keys():                               # [0,1,2,3,4]
        pred_poses_quat = return_dict[it]['poses_cam']          # [b*(t-1),7], b=1
        pred_poses_mat = geo_utils.quat2mat(pred_poses_quat)    # [b*(t-1),4,4]
        permutation = return_dict[it]['permutation']
        assert it == str(permutation[0])
        for idx in range(t-1):
            pose_dict[(int(it), permutation[idx+1])] = pred_poses_mat[idx].to(device) 
            if str(best_canonical_id) == it:
                best_canonical_pairs.append((int(it), idx))

    # calculate confidence by T @ T.inv() = I
    for idx1 in return_dict.keys():
        for idx2 in range(t):
            if idx1 == str(idx2):
                conf_dict[(int(idx1), int(idx2))] = 1.0
            else:
                pose1 = pose_dict[(int(idx1), int(idx2))]  # [4,4]
                pose2 = pose_dict[(int(idx2), int(idx1))]
                tmp = geo_utils.mat2quat((pose1 @ pose2).unsqueeze(0)).cpu().squeeze()
                tmp_I = geo_utils.mat2quat(torch.eye(4).unsqueeze(0)).cpu().squeeze()
                theta, trans = eval_utils.compute_pose_metric(tmp, tmp_I)    # theta in degree
                conf_pose = (np.cos(theta * np.pi / 180.) + 1) / 2
                conf_dict[(int(idx1), int(idx2))] = torch.tensor([float(conf_pose)]).to(device)

    # get input of sync, pose_dict saves relative pose, turn it into relative extrinsics
    Ps, confidence = {}, {}
    target_pairs = list(combinations(range(t), 2))
    for pair in target_pairs:
        confidence[pair] = conf_dict[pair]
        if pair in best_canonical_pairs:
            Ps[pair] = torch.inverse(pose_dict[pair].unsqueeze(0))      # pose to extrinsics
        elif pair[::-1] in best_canonical_pairs:
            Ps[pair] = pose_dict[pair[::-1]].unsqueeze(0)
        else:
            Ps[pair] = torch.inverse(pose_dict[pair].unsqueeze(0))
    
    # sync
    Ps_sync = sync_utils.camera_synchronization(Ps, confidence, N=t, squares=10, center_first_camera=True)  # [b=1,t,4,4]

    # get relative poses with canonical view index best_canonical_id
    poses = torch.inverse(Ps_sync[0])   # [t,4,4]
    poses = poses[return_dict[str(best_canonical_id)]['permutation']]
    poses_rel = geo_utils.get_relative_pose(poses[0], poses[1:])
    poses_rel_quat = geo_utils.mat2quat(poses_rel).to(device)
    return poses_rel_quat


def get_all_combinations(n_views):
    views = [it for it in range(n_views)]

    all_combinations = []
    for i in range(1, n_views+1):
        cur_combination = list(list(it) for it in combinations(views, i))
        all_combinations += cur_combination
    return all_combinations


def visualize_360_all(model, sample, dataset, poses_cam, features, batch_idx, name, output_dir, device):
    b, t, C, D, H, W = features.shape

    all_combinations = get_all_combinations(n_views=t)
    for it in all_combinations:
        visualize_360(model, sample, dataset, poses_cam, features, batch_idx, name, output_dir, device, combination=it)


def visualize_360(model, sample, dataset, poses_cam, features, batch_idx, name, output_dir, device, combination=None):
    b, t, C, D, H, W = features.shape

    # get camera extrinsics and pose   
    camPoseRel_cv2 = model.module.encoder_traj.toSE3(poses_cam).to(device)
    canonical_pose_cv2 = dataset.get_canonical_pose_cv2(device=device)                                  # [4,4]
    canonical_extrinsics_cv2 = dataset.get_canonical_extrinsics_cv2(device=device)
    camPoses_cv2 = canonical_pose_cv2.unsqueeze(0) @ camPoseRel_cv2
    camE_cv2 = torch.inverse(camPoses_cv2)                                                              # [b*(t-1),4,4], canonicalized extrinsics
    camE_cv2 = camE_cv2.reshape(b,t-1,4,4)
    camPoses_cv2 = camPoses_cv2.reshape(b,t-1,4,4)
    camPoses_cv2 = torch.cat([canonical_pose_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camPoses_cv2], dim=1)# [b,t,4,4]
    camE_cv2 = torch.cat([canonical_extrinsics_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camE_cv2], dim=1)  # [b,t,4,4]

    if combination is not None:
        print('combination', combination)
        combination = [0] + combination
        features = features[:,combination]
        camPoses_cv2 = camPoses_cv2[:,combination]
        img_name = 'sample{}_{}views_{}'.format(batch_idx, len(combination)-1, '_'.join([str(num) for num in combination[1:]]))
        t = len(combination)
    else:
        img_name = str(batch_idx)

    # sample NVS camera poses
    num_views_all = 4 * 7
    elev = torch.linspace(0, 0, num_views_all)
    azim = torch.linspace(0, 360, num_views_all) + 180
    NVS_R_all, NVS_T_all = look_at_view_transform(dist=config.render.camera_z, elev=elev, azim=azim)  # [N,3,3], [N,3]
    NVS_pose_all = torch.cat([NVS_R_all, NVS_T_all.view(-1,3,1)], dim=-1)  # [N,3,4]

    # feature fusion
    features_transformed = model.module.rotate(voxels=features, camPoses_cv2=camPoses_cv2[:,:t], grid_size=D)  # [b,t,C,D,H,W]
    if combination is not None:
        features_transformed = features_transformed[:,1:]
        camPoses_cv2 = camPoses_cv2[:,1:]
    idxs = sequence_from_distance(camPoses_cv2[:,:,:3,3])
    features_transformed = chose_selected(features_transformed, idxs)
    features_mv = model.module.encoder_3d.fuse(features_transformed)            # [b,t,C,D,H,W] -> [b,C,D,H,W]
    densities_mv = model.module.encoder_3d.get_density3D(features_mv)           # [b,1,D,H,W]
    features_mv = model.module.encoder_3d.get_render_features(features_mv)      # [b,C,D,H,W]

    # 360-degree NVS
    for idx in range(b):
        rendered_imgs_results, rendered_masks_results, rendered_depths_results = [], [], []
        all_feature = features_mv[idx].unsqueeze(0).repeat(7,1,1,1,1)                   # [N,C,D,H,W]
        all_density = densities_mv[idx].unsqueeze(0).repeat(7,1,1,1,1).clamp(max=1.0)   # [N,1,D,H,W]
        for pose_idx in range(4):
            cameras = {
                'K': sample['K_cv2'][idx][0:1].repeat(7,1,1),                      # [N,3,3]
                'R': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,:3],
                'T': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,3],
            }
            rendered_imgs, rendered_masks, rendered_depths = model.module.render(cameras, all_feature, all_density, render_depth=True)           # [N,c,h,w], [N,1,h,w]
            rendered_imgs_results.append(rendered_imgs.detach())
            rendered_masks_results.append(rendered_masks.detach())
            rendered_depths_results.append(rendered_depths.detach())
        rendered_imgs_results = torch.cat(rendered_imgs_results, dim=0)
        rendered_masks_results = torch.cat(rendered_masks_results, dim=0)
        rendered_depths_results = torch.cat(rendered_depths_results, dim=0)
        vis_utils.vis_NVS(imgs=rendered_imgs_results,
                            masks=rendered_masks_results, 
                            img_name=img_name + '_' + str(idx),
                            output_dir=output_dir,
                            subfolder=os.path.join('vis_360', name),
                            depths=rendered_depths_results
                         )


def evaluate_all(model, lpips_vgg, sample, dataset, return_dict, batch_idx, device, output_dir, name='before', eval_pose=True):
    eval_results = {}

    for canonical_id in range(5):
        cur_results = return_dict[str(canonical_id)]
        poses_cam, features, nvs_extr, gt_poses = cur_results['poses_cam'], cur_results['features_raw'], cur_results['nvs_extr'], cur_results['gt_poses']
        psnr, ssim, lpips, rot_error, trans_error, depth_error = evaluate(model, lpips_vgg, sample, dataset, poses_cam, features, nvs_extr, gt_poses, batch_idx, canonical_id, device, output_dir, 'before')
        eval_results[str(canonical_id)] = {
            'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 
            'rot_error': rot_error, 'trans_error': trans_error, 'depths_error': depth_error,
            'gt_poses': gt_poses}
    
    psnr_res = [(eval_results[k]['psnr'], k) for k in eval_results.keys()]
    psnr_res = sorted(psnr_res, key = lambda x: x[0], reverse=True)
    rot_error_res = [(eval_results[k]['rot_error'], k) for k in eval_results.keys()]
    rot_error_res = sorted(rot_error_res, key = lambda x: x[0], reverse=True)
    best_canonical_id = rot_error_res[-1][1] #psnr_res[0][1]

    results = eval_results[str(best_canonical_id)]
    psnr, ssim, lpips, rot_error, trans_error = results['psnr'], results['ssim'], results['lpips'], results['rot_error'], results['trans_error']
    return best_canonical_id, psnr, ssim, lpips, rot_error, trans_error, depth_error


def evaluate(model, lpips_vgg, sample, dataset, poses_cam, features, nvs_extr, gt_poses, batch_idx, canonical_id, device, output_dir, name='before', eval_pose=True):
    b, t, C, D, H, W = features.shape
    
    # get camera extrinsics and pose
    camPoseRel_cv2 = model.module.encoder_traj.toSE3(poses_cam).to(device)
    canonical_pose_cv2 = dataset.get_canonical_pose_cv2(device=device)                                      # [4,4]
    canonical_extrinsics_cv2 = dataset.get_canonical_extrinsics_cv2(device=device)
    camPoses_cv2 = canonical_pose_cv2.unsqueeze(0) @ camPoseRel_cv2
    camE_cv2 = torch.inverse(camPoses_cv2)                                                                  # [b*(t-1),4,4], canonicalized extrinsics
    camE_cv2 = camE_cv2.reshape(b,t-1,4,4)
    camPoses_cv2 = camPoses_cv2.reshape(b,t-1,4,4)
    camPoses_cv2 = torch.cat([canonical_pose_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camPoses_cv2], dim=1)    # [b,t,4,4]
    camE_cv2 = torch.cat([canonical_extrinsics_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camE_cv2], dim=1)      # [b,t,4,4]
    
    # evaluate NVS    
    clips = sample['images'].to(device)
    clips_nvs = clips[:,5:]
    #depths_nvs = sample['depths'][:,:5].to(device)
    b, t2, c, h, w = clips_nvs.shape
    
    features_transformed = model.module.rotate(voxels=features, camPoses_cv2=camPoses_cv2[:,:t], grid_size=D)     # [b,t,C,D,H,W]
    idxs = sequence_from_distance(camPoses_cv2[:,:,:3,3])
    features_transformed = chose_selected(features_transformed, idxs)
    
    features_mv = model.module.encoder_3d.fuse(features_transformed)                        # [b,t,C,D,H,W] -> [b,C,D,H,W]
    densities_mv = model.module.encoder_3d.get_density3D(features_mv)                       # [b,1,D,H,W]
    features_mv = model.module.encoder_3d.get_render_features(features_mv)                  # [b,C,D,H,W]

    # visualize NVS
    cameras = {
                    'R': nvs_extr[:,5:].reshape(b*5,4,4)[:,:3,:3].to(device),                # [b*t,3,3]
                    'T': nvs_extr[:,5:].reshape(b*5,4,4)[:,:3,3].to(device),                 # [b*t,3]
                    'K': sample['K_cv2'][:,5:].reshape(b*5,3,3).to(device)                   # [b*t,3,3]
                }

    _, C2, D2, H2, W2 = features_mv.shape
    features_all = features_mv.unsqueeze(1).repeat(1,t2,1,1,1,1).reshape(b*t2,C2,D2,H2,W2)   # [b,2*t,C,D,H,W] -> [b*2*t,C,D,H,W]
    densities_all = densities_mv.unsqueeze(1).repeat(1,t2,1,1,1,1).reshape(b*t2,1,D2,H2,W2)
    #rendered_imgs, rendered_masks, rendered_depths = model.module.render(cameras, features_all, densities_all, render_depth=True)
    rendered_imgs, rendered_masks = model.module.render(cameras, features_all, densities_all, render_depth=False)
    rendered_imgs = rendered_imgs.reshape(b,t2,c,h,w)
    rendered_masks = rendered_masks.reshape(b,t2,1,h,w)
    #rendered_depths = rendered_depths.reshape(b,t2,1,h,w)

    psnr, ssim = 0.0, 0.0
    for seq_idx in range(5):
        cur_img_recon = rendered_imgs[0, seq_idx].permute(1,2,0).detach().cpu().numpy()
        cur_img_gt = clips_nvs[0, seq_idx].permute(1,2,0).cpu().numpy()
        cur_psnr, cur_ssim = eval_utils.compute_img_metric(cur_img_recon, cur_img_gt)
        psnr += cur_psnr
        ssim += cur_ssim
    psnr /= 5.0
    ssim /= 5.0
    lpips = lpips_vgg(rendered_imgs[0], clips_nvs[0]).mean().item()
    #depth_error = torch.clamp(torch.abs(depths_nvs - rendered_depths).mean(), min=0.0, max=2.0).item()
    
    vis_utils.vis_seq(vid_clips=sample['images'][:,5:],
                        vid_masks=sample['fg_probabilities'][:,5:],
                        recon_clips=rendered_imgs,
                        recon_masks=rendered_masks,
                        iter_num=str(batch_idx)+'_'+str(canonical_id),
                        output_dir=output_dir,
                        subfolder=os.path.join('nvs', name),
                        #vid_depths=sample['depths'][:,5:],
                        #recon_depths=rendered_depths
                    )

    # visualize inputs
    cameras = {
                    'R': camE_cv2.reshape(-1,4,4)[:,:3,:3].to(device),                  # [b*t,3,3]
                    'T': camE_cv2.reshape(-1,4,4)[:,:3,3].to(device),                   # [b*t,3]
                    'K': sample['K_cv2'][:,:5].reshape(b*5,3,3).to(device)              # [b*t,3,3]
                }
    rendered_imgs, rendered_masks, rendered_depths = model.module.render(cameras, features_all, densities_all, render_depth=True)
    rendered_imgs = rendered_imgs.reshape(b,t2,c,h,w)
    rendered_masks = rendered_masks.reshape(b,t2,1,h,w)
    rendered_depths = rendered_depths.reshape(b,t2,1,h,w)
    permute = [int(canonical_id)] + [it for it in range(t) if it != int(canonical_id)]
    vid_clips = sample['images'][:,:5][:,permute]
    vid_masks = sample['fg_probabilities'][:,:5][:,permute]
    #vid_depths = sample['depths'][:,:5][:,permute]
    vis_utils.vis_seq(vid_clips=vid_clips,
                        vid_masks=vid_masks,
                        recon_clips=rendered_imgs,
                        recon_masks=rendered_masks,
                        iter_num=str(batch_idx)+'_'+str(canonical_id),
                        output_dir=output_dir,
                        subfolder=os.path.join('inputs', name),
                        #vid_depths=vid_depths,
                        #recon_depths=rendered_depths
                    )    
    
    depth_error = 0.0
    if eval_pose == False:
        print('Batch {}, canonical_id {}, {}, psnr: {}, ssim: {}, lpips: {}, depth {}'.format(batch_idx, canonical_id, name, psnr, ssim, lpips, depth_error))
        print('----------------------------------------------------------------')
        return psnr, ssim, lpips, 0.0#, depth_error
    
    # evaluate pose
    poses_cam_gt = gt_poses[:,1:5].to(device).reshape(b*(t-1),4,4) 
    poses_cam_gt = geo_utils.mat2quat(poses_cam_gt)
    camPose_return = {'gt': poses_cam_gt, 'pred': poses_cam}
    rot_error, trans_error = 0.0, 0.0
    for img_idx in range(4):
        cur_rot_error, cur_trans_error = eval_utils.compute_pose_metric(camPose_return['pred'][img_idx].detach().cpu(), 
                                                            camPose_return['gt'][img_idx].detach().cpu())
        rot_error += cur_rot_error if cur_rot_error < 50 else 50
        trans_error += cur_trans_error
    rot_error /= 5.0
    trans_error /= 5.0
    print('Batch {}, canonical_id {}, {}, psnr: {}, ssim: {}, lpips: {}, rot_error: {}, trans_error: {}, depth_error {}'.format(batch_idx, canonical_id, name, psnr, ssim, lpips, rot_error, trans_error, depth_error))
    return psnr, ssim, lpips, rot_error, trans_error, 0.0#, depth_error



def predict_initial(model, sample, device):
    return_dict = {}

    for canonical_id in range(5):
        clips = sample['images'].to(device)
        clips_nvs = clips[:,5:]
        clips = clips[:,:5]
        K_cv2 = sample['K_cv2'].to(device)
        K_cv2 = K_cv2[:,:5]
        b, t, c, h, w = clips.shape

        gt_poses = sample['cam_poses_rel_cv2'][:,:5]
        nvs_extr = sample['cam_extrinsics_cv2_canonicalized']
        clips, gt_poses, nvs_extr, permute = permute_clips(clips, gt_poses, nvs_extr, canonical_id, return_permutation=True)

        clips = clips.reshape(b*t,c,h,w)
        features_raw = model.module.encoder_3d.get_feat3D(clips)                        # [b*t,C,D,H,W]
        _, C, D, H, W = features_raw.shape
        clips = clips.reshape(b,t,c,h,w)
        features_raw = features_raw.reshape(b,t,C,D,H,W)

        pose_feat_3d = model.module.encoder_traj(features_raw, return_features=True)    # [b*(t-1),1024]
        pose_feat_2d = model.module.encoder_traj_2d(clips, return_features=True)        # [b*(t-1),1024]
        pose_feat = torch.cat([pose_feat_3d, pose_feat_2d], dim=-1)                     # [b*(t-1),2048]
        pred = model.module.pose_head(pose_feat)                                        # [b*(t-1), 8]
        poses_cam, conf = pred.split([model.module.encoder_traj.pose_dim, 1], dim=-1)
        tmp = torch.zeros_like(poses_cam)
        tmp[:,:4] = F.normalize(poses_cam[:,:4])
        tmp[:,4:] = poses_cam[:,4:]
        poses_cam = tmp

        return_dict[str(canonical_id)] = {'permutation': permute,
                                          'poses_cam': poses_cam.detach(),          # [b*(t-1), 7]
                                          #'poses_cam': geo_utils.mat2quat(gt_poses[:,1:].reshape(-1, 4, 4)).to(device),
                                          'features_raw': features_raw.detach(),    # [b,t,C,D,H,W]
                                          'nvs_extr': nvs_extr,
                                          'gt_poses': gt_poses}

    return return_dict


def do_refinement(batch_idx, model, sample, dataset, poses_cam_all, features_all, gt_poses_all, 
                  clips, masks, device, canonical_id, chosen_idx=[0,1,2,3,4], iter_num=500):
    '''
    poses_cam_all: in [b*(t-1),7]
    features_all: in [b,t,C,D,H,W]
    gt_poses_all: in [b,t,4,4]
    '''
    pose_chosen_idx = []
    for it in chosen_idx:
        if it != 0:
            pose_chosen_idx.append(it-1)
    gt_pose_chosen_idx = [it + 1 for it in pose_chosen_idx]
    t = len(chosen_idx)

    b, t_all, c, h, w = clips.shape
    _, _, C, D, H, W = features_all.shape
    poses_cam_all = poses_cam_all.reshape(b,-1,7)

    features = features_all[:, chosen_idx]
    target_imgs = clips[:, chosen_idx].view(-1,c,h,w)
    target_masks = masks[:, chosen_idx].view(-1,1,h,w)
    gt_poses = gt_poses_all[:, gt_pose_chosen_idx].view(-1,4,4)

    poses_cam = poses_cam_all[:, pose_chosen_idx].view(-1,7).detach()
    #print(gt_poses.shape, poses_cam.shape)
    poses_cam.requires_grad = True
    poses_cam_rot = poses_cam[:,:4].detach()
    poses_cam_trans = poses_cam[:,4:].detach()
    poses_cam_rot.requires_grad = True
    poses_cam_trans.requires_grad = True
    lr_start, lr_end = 0.001, 0.001
    optimizer = torch.optim.Adam([{'params': poses_cam_rot, 'lr': lr_start},
                                 {'params': poses_cam_trans, 'lr': lr_start / 2.0}
                                ], lr=lr_start)
    gamma = (lr_end / lr_start) ** (1.0 / iter_num) 
    schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    
    batch_end = time.time()
    for iter_idx in range(iter_num+1):
        # quaternion to rot mat
        poses_cam_normalized = torch.zeros_like(poses_cam).to(device)
        poses_cam_normalized[:,:4] = F.normalize(poses_cam_rot)
        poses_cam_normalized[:,4:] = poses_cam_trans

        camPoseRel_cv2 = model.module.encoder_traj.toSE3(poses_cam_normalized).to(device)     # [b*(t-1),4,4]

        # get camera extrinsics and pose for rendering
        canonical_pose_cv2 = dataset.get_canonical_pose_cv2(device=device)                                  # [4,4]
        canonical_extrinsics_cv2 = dataset.get_canonical_extrinsics_cv2(device=device)
        camPoses_cv2 = canonical_pose_cv2.unsqueeze(0) @ camPoseRel_cv2
        camE_cv2 = torch.inverse(camPoses_cv2)                                                              # [b*(t-1),4,4], canonicalized extrinsics
        camE_cv2 = camE_cv2.reshape(b,t-1,4,4)
        camPoses_cv2 = camPoses_cv2.reshape(b,t-1,4,4)
        camPoses_cv2 = torch.cat([canonical_pose_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camPoses_cv2], dim=1)
        camE_cv2 = torch.cat([canonical_extrinsics_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camE_cv2], dim=1)

        # transform features
        features_transformed = model.module.rotate(voxels=features, camPoses_cv2=camPoses_cv2[:,:t], grid_size=D)     # [b,t,C,D,H,W]
        idxs = sequence_from_distance(camPoses_cv2[:,:,:3,3])
        features_transformed = chose_selected(features_transformed, idxs)

        features_mv = model.module.encoder_3d.fuse(features_transformed)  # [b,t,C=128,D=16,H,W] -> [b,C,D,H,W]
        densities_mv = model.module.encoder_3d.get_density3D(features_mv)  # [b,1,D=32,H,W]
        features_mv = model.module.encoder_3d.get_render_features(features_mv)    # [b,C=16,D=32,H,W]
        _, C2, D2, H2, W2 = features_mv.shape

        # render
        camE_cv2 = camE_cv2.repeat(1,1,1,1)             # [b,2*t,4,4]
        camPoses_cv2 = camPoses_cv2.repeat(1,1,1,1)     # [b,2*t,4,4]
        camK = sample['K_cv2'][:,:t].repeat(1,1,1,1)          # [b,2*t,3,3]
        cameras = {
                'R': camE_cv2.reshape(b*1*t,4,4)[:,:3,:3].to(device),                # [b*t,3,3]
                'T': camE_cv2.reshape(b*1*t,4,4)[:,:3,3].to(device),                 # [b*t,3]
                'K': camK.reshape(b*1*t,3,3).to(device)                   # [b*t,3,3]
            }
        
        features_all = features_mv.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t,C2,D2,H2,W2)    # [b,2*t,C,D,H,W] -> [b*2*t,C,D,H,W]
        densities_all = densities_mv.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t,1,D2,H2,W2)
        rendered_imgs, rendered_masks, rendered_depths, origin_proj = model.module.render(cameras, features_all, densities_all, 
                                                                                            return_origin_proj=True, render_depth=True)

        # calculate loss
        origin_proj = 2.0 * origin_proj / config.dataset.img_size
        loss_recon_img = F.mse_loss(rendered_imgs, target_imgs)
        loss_recon_mask = F.mse_loss(rendered_masks, target_masks)
        loss_regu_origin = F.mse_loss(origin_proj, 0.5 * torch.ones_like(origin_proj).to(device))
        loss_recon = config.loss.recon_rgb * loss_recon_img + config.loss.recon_mask * loss_recon_mask

        # optimize pose
        optimizer.zero_grad()
        loss_recon.backward()
        optimizer.step()
        schedular.step()

        # print information
        poses_cam_gt = gt_poses.to(device).reshape(b*(t-1),4,4)
        poses_cam_gt = geo_utils.mat2quat(poses_cam_gt)
        camPose_return = {'gt': poses_cam_gt, 'pred': poses_cam_normalized}
        rot_error, trans_error = 0.0, 0.0
        for img_idx in range(len(pose_chosen_idx)):
            cur_rot_error, cur_trans_error = eval_utils.compute_pose_metric(camPose_return['pred'][img_idx].detach().cpu(), 
                                                                camPose_return['gt'][img_idx].detach().cpu())
            rot_error += cur_rot_error if cur_rot_error < 50 else 50
            trans_error += cur_trans_error
        rot_error /= len(chosen_idx)
        trans_error /= len(chosen_idx)

        if iter_idx % 500 == 0:
            info = '-- Batch {}, canonical_id {}, chosen_idx {}, iter {}, time {:.3f}, rot error {:.5f}, trans error {:.5f}, rgb loss {:.5f}, mask loss {:5f}'.format(
                batch_idx, canonical_id, chosen_idx, iter_idx, time.time() - batch_end,
                rot_error, trans_error, loss_recon_img.item(), loss_recon_mask.item()
            )
            if config.loss.regu_origin_proj > 0:
                info += ', origin regu loss {:.5f}'.format(loss_regu_origin.item())
            print(info)
        batch_end = time.time()

    poses_cam_all[:, pose_chosen_idx] = poses_cam.detach()   # [b,t-1,7]
    return poses_cam_all.view(-1,7), rot_error, trans_error, camPoses_cv2


def refine_pose(batch_idx, model, sample, dataset, poses_cam, features, canonical_id, device, output_dir, vis=True, iter_num=5000):
    clips = sample['images'].to(device)
    clips = clips[:,:5]
    masks = sample['fg_probabilities'].to(device)
    masks = masks[:,:5]
    K_cv2 = sample['K_cv2'].to(device)
    K_cv2 = K_cv2[:,:5]
    b, t, c, h, w = clips.shape
    _, _, C, D, H, W = features.shape

    gt_poses = sample['cam_poses_rel_cv2'][:,:5]
    nvs_extr = sample['cam_extrinsics_cv2_canonicalized']
    clips, gt_poses, nvs_extr = permute_clips(clips, gt_poses, nvs_extr, canonical_id)
    masks = permute_clips(masks, None, None, canonical_id, clips_only=True)

    _, _, _, camPoses_cv2_initial = do_refinement(batch_idx, model, sample, dataset, poses_cam, features, gt_poses, 
                                  clips, masks, device, canonical_id, chosen_idx=[0,1,2,3,4], iter_num=1)
    
    poses_cam, rot_error, trans_error, camPoses_cv2 = do_refinement(batch_idx, model, sample, dataset, poses_cam, features, gt_poses, 
                                  clips, masks, device, canonical_id, chosen_idx=[0,1,2,3,4], iter_num=iter_num)

    gt_poses[:,:,2,3] -= 4.0
    if vis:
        vis_utils.vis_poses(clips, camPoses_cv2_initial, gt_poses, output_dir, os.path.join('before', str(batch_idx)))
        vis_utils.vis_poses(clips, camPoses_cv2, gt_poses, output_dir, os.path.join('after', str(batch_idx)))
    
    poses_final = torch.zeros_like(poses_cam)
    poses_cam_rot = poses_cam[:,:4].detach()
    poses_cam_trans = poses_cam[:,4:].detach()
    poses_final[:,:4] = F.normalize(poses_cam_rot)
    poses_final[:,4:] = poses_cam_trans
    return poses_final, rot_error, trans_error



def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FORGE')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        "--sync", dest="sync", action="store_true", help="whether use camera synchronization")
    parser.add_argument(
        "--split_num", help='number of experiments', default=8, type=int)
    parser.add_argument(
        "--exp_id", help='experiment name for head-craft multithreading', default=0, type=int)
    parser.add_argument(
        "--iter_num", help='number of opmization iteration, generally 1000 is already good enough', default=5000, type=int)
    parser.add_argument(
        "--model_gt", dest="model_gt", action="store_true",help="use un-degenerated fusion module")
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    # Get args and config
    args = parse_args()
    logger, output_dir, _ = exp_utils.create_logger(config, args.cfg, phase='train')
    print('=> Saving args and config into logger...')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # set device
    gpus = range(torch.cuda.device_count())
    device = torch.device('cuda') if len(gpus) > 0 else torch.device('cpu')

    # get model
    model = ReconModel(config).to(device)
    cpt_root = './output/omniobject3d/joint_pose_2d3d/pred_pose_2d3d_joint'
    cpt_name = 'cpt_best_psnr_25.58508383186279_rot_8.334599444377726.pth.tar'
    cpt = torch.load(os.path.join(cpt_root, cpt_name))['state_dict']
    model.load_state_dict(cpt, strict=True)
    model = torch.nn.DataParallel(model)
    
    # get model trained with gt pose
    model_gt = ReconModel(config).to(device)
    cpt_root = './output/omniobject3d/gt_pose/gt_pose'
    cpt_name = 'cpt_last.pth.tar'
    cpt = torch.load(os.path.join(cpt_root, cpt_name))['state_dict']
    del cpt['encoder_traj.out.3.weight']
    del cpt['encoder_traj.out.3.bias']
    model_gt.load_state_dict(cpt, strict=False)
    model_gt = torch.nn.DataParallel(model_gt)

    lpips_vgg = lpips.LPIPS(net="vgg").to(device)
    lpips_vgg.eval()
    

    val_data = Omniobject3D(config, split='test')
    val_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=config.test.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=False)    

    run_optimization(args, config, loader=val_loader, dataset=val_data, model=model, model_gt=model_gt, lpips_vgg=lpips_vgg, output_dir=output_dir, device=device)


if __name__ == '__main__':
    main()