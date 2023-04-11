import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
import os
import dataclasses
import torch.distributed as dist
import itertools
from utils import train_utils, vis_utils, geo_utils, exp_utils, eval_utils
from pytorch3d.renderer import look_at_view_transform


def validate_poseEstimator3D(config, loader, dataset, model, epoch, output_dir, device, rank):
    '''
    validation for only using 3D pose estimator
    '''
    model.eval()
    mode = config.train.parameter
    assert mode in ['all', 'pose', 'joint']
    # 'all' for training model using GT pose, return psnr only, (rot=0)
    # 'pose' for training pose estimator only, return rot only, (psnr=float('inf'))
    # 'joint' for joint training, return both
 
    unseen_psnr = []
    unseen_ssim = []
    seen_psnr = []
    seen_ssim = []
    unseen_rot = []
    unseen_trans = []
    seen_rot = []
    seen_trans = []

    # sample NVS camera poses
    num_views_all = 4 * 7
    if not config.train.canonicalize:
        elev = torch.linspace(0, 360, num_views_all)
        azim = torch.linspace(0, 0, num_views_all) + 180
    else:
        elev = torch.linspace(0, 0, num_views_all)
        azim = torch.linspace(0, 360, num_views_all) + 180
    NVS_R_all, NVS_T_all = look_at_view_transform(dist=config.render.camera_z, elev=elev, azim=azim)  # [N,3,3], [N,3]
    NVS_pose_all = torch.cat([NVS_R_all, NVS_T_all.view(-1,3,1)], dim=-1)  # [N,3,4]

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            if batch_idx % config.eval_vis_freq != 0:
                continue

            clips = sample['images'].to(device)
            clips_nvs = clips[:,5:]
            clips = clips[:,:5]
            masks = sample['fg_probabilities'].to(device)
            masks = masks[:,:5]
            K_cv2 = sample['K_cv2'].to(device)
            K_cv2 = K_cv2[:,:5]
            b, t, c, h, w = clips.shape
            seen_flag = sample['seen_flag']
            cur_seen_flag = seen_flag[0].item() > 0     # batch=1
            
            clips = clips.reshape(b*t,c,h,w)
            features_raw = model.module.encoder_3d.get_feat3D(clips)               # [b*t,C,D,H,W]
            _, C, D, H, W = features_raw.shape
            clips = clips.reshape(b,t,c,h,w)
            features_raw = features_raw.reshape(b,t,C,D,H,W)
            
            if not config.train.use_gt_pose:
                # predict relative poses
                poses_cam, conf = model.module.encoder_traj(features_raw)
                tmp = torch.zeros_like(poses_cam)
                tmp[:,:4] = F.normalize(poses_cam[:,:4])
                tmp[:,4:] = poses_cam[:,4:]
                poses_cam = tmp
                camPoseRel_cv2 = model.module.encoder_traj.toSE3(poses_cam)                                               # [b*(t-1),4,4], relative cam pose in cv2 frame
                
                # get camera extrinsics and pose
                canonical_pose_cv2 = dataset.get_canonical_pose_cv2(device=device)                                  # [4,4]
                canonical_extrinsics_cv2 = dataset.get_canonical_extrinsics_cv2(device=device)
                camPoses_cv2 = canonical_pose_cv2.unsqueeze(0) @ camPoseRel_cv2
                camE_cv2 = torch.inverse(camPoses_cv2)                                                              # [b*(t-1),4,4], canonicalized extrinsics
                camE_cv2 = camE_cv2.reshape(b,t-1,4,4)
                camPoses_cv2 = camPoses_cv2.reshape(b,t-1,4,4)
                camPoses_cv2 = torch.cat([canonical_pose_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camPoses_cv2], dim=1)
                camE_cv2 = torch.cat([canonical_extrinsics_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camE_cv2], dim=1)  # [b,t,4,4]
                poses_cam_gt = sample['cam_poses_rel_cv2'][:,1:5].to(device).reshape(b*(t-1),4,4)
                poses_cam_gt = geo_utils.mat2quat(poses_cam_gt)
                camPose_return = {'gt': poses_cam_gt, 'pred': poses_cam}
            else:
                # use ground truth poses
                camE_cv2 = sample['cam_extrinsics_cv2_canonicalized'][:,:5].to(device)     # [b,t,4,4]
                camPoses_cv2 = sample['cam_poses_cv2_canonicalized'][:,:5].to(device)
                poses_cam_gt = sample['cam_poses_rel_cv2'][:,1:5].to(device).reshape(b*(t-1),4,4)
                poses_cam_gt = geo_utils.mat2quat(poses_cam_gt)
                camPose_return = {'gt': poses_cam_gt, 'pred': poses_cam_gt}
            
            # parse camera for rendering using first five views
            camE_cv2 = camE_cv2.repeat(1,2,1,1)             # [b,2*t,4,4]
            camPoses_cv2 = camPoses_cv2.repeat(1,2,1,1)     # [b,2*t,4,4]
            camK = sample['K_cv2'][:,:5].repeat(1,2,1,1)          # [b,2*t,3,3]
            cameras = {
                'R': camE_cv2.reshape(b*2*t,4,4)[:,:3,:3],                # [b*2t,3,3]
                'T': camE_cv2.reshape(b*2*t,4,4)[:,:3,3],                 # [b*2t,3]
                'K': camK.reshape(b*2*t,3,3).to(device)                   # [b*2t,3,3]
            }

            if mode != 'pose':
                # transform features using camera poses
                features_transformed = model.module.rotate(voxels=features_raw, camPoses_cv2=camPoses_cv2[:,:t], grid_size=D)   # [b,t,C=128,D=16,H,W]

                # get prediction using features of some views
                # 1) features of first 3 views and render using last two cams
                # 2) features of last 2 views and render using first 3 cams
                features_3v = model.module.encoder_3d.fuse(features_transformed[:,:3])    # [b,C=128,D=16,H,W]
                features_2v = model.module.encoder_3d.fuse(features_transformed[:,-2:])
                densities_3v2v = model.module.encoder_3d.get_density3D(torch.cat([features_3v, features_2v], dim=0))    # [2*b,C=1,D=32,H,W]
                features_3v2v = model.module.encoder_3d.get_render_features(torch.cat([features_3v, features_2v], dim=0))  # [2*b,C=16,D=32,H,W]
                features_3v = features_3v2v[:b].unsqueeze(1).repeat(1,2,1,1,1,1)    # [b,2,C=16,D=32,H,W]
                features_2v = features_3v2v[b:].unsqueeze(1).repeat(1,3,1,1,1,1)    # [b,3,C=16,D=32,H,W]
                features_3v2v = torch.cat([features_2v, features_3v], dim=1)    # [b,t,C=16,D=32,H,W]
                densities_3v = densities_3v2v[:b].unsqueeze(1).repeat(1,2,1,1,1,1)  # [b,2,1,D,H,W]
                densities_2v = densities_3v2v[b:].unsqueeze(1).repeat(1,3,1,1,1,1)
                densities_3v2v = torch.cat([densities_2v, densities_3v], dim=1) # [b,t,1,D,H,W]
                
                # get prediction using multi-view information
                cur_time = time.time()
                features_mv = model.module.encoder_3d.fuse(features_transformed)  # [b,t,C=128,D=16,H,W] -> [b,C,D,H,W]
                densities_mv = model.module.encoder_3d.get_density3D(features_mv)  # [b,1,D=32,H,W]
                features_mv = model.module.encoder_3d.get_render_features(features_mv)    # [b,C=16,D=32,H,W]
                _, C2, D2, H2, W2 = features_mv.shape
                features_all = features_mv.unsqueeze(1).repeat(1,t,1,1,1,1)
                features_all = torch.cat([features_3v2v, features_all], dim=1).reshape(b*2*t,C2,D2,H2,W2)    # [b,2*t,C,D,H,W] -> [b*2*t,C,D,H,W]
                densities_all = densities_mv.unsqueeze(1).repeat(1,t,1,1,1,1)
                densities_all = torch.cat([densities_3v2v, densities_all], dim=1).reshape(b*2*t,1,D2,H2,W2)

                rendered_imgs, rendered_masks, origin_proj = model.module.render(cameras, features_all, densities_all, return_origin_proj=True)
                rendered_imgs = rendered_imgs.reshape(b,2*t,c,h,w)
                rendered_masks = rendered_masks.reshape(b,2*t,1,h,w)

                # parse cameras for rendering using novel views
                cameras_nvs = {
                    'R': sample['cam_extrinsics_cv2_canonicalized'][:,5:].reshape(b*5,4,4)[:,:3,:3].to(device),                # [b*t,3,3]
                    'T': sample['cam_extrinsics_cv2_canonicalized'][:,5:].reshape(b*5,4,4)[:,:3,3].to(device),                 # [b*t,3]
                    'K': sample['K_cv2'][:,5:].reshape(b*5,3,3).to(device)                   # [b*t,3,3]
                }
                features_all = features_mv.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t,C2,D2,H2,W2)
                densities_all = densities_mv.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t,1,D2,H2,W2)
                rendered_imgs_nvs, rendered_masks_nvs = model.module.render(cameras_nvs, features_all, densities_all, return_origin_proj=False)
                rendered_imgs_nvs = rendered_imgs_nvs.reshape(b,5,c,h,w)
                rendered_masks_nvs = rendered_masks_nvs.reshape(b,5,1,h,w)

                # evaluate
                rendered_imgs = rendered_imgs.clip(min=0.0, max=1.0)
                rendered_imgs_nvs = rendered_imgs_nvs.clip(min=0.0, max=1.0)

                psnr, ssim = 0.0, 0.0
                for seq_idx in range(5):
                    cur_img_recon = rendered_imgs_nvs[0, seq_idx].permute(1,2,0).detach().cpu().numpy()
                    cur_img_gt = clips_nvs[0, seq_idx].permute(1,2,0).cpu().numpy()
                    cur_psnr, cur_ssim = eval_utils.compute_img_metric(cur_img_recon, cur_img_gt)
                    psnr += cur_psnr
                    ssim += cur_ssim
                psnr /= 5.0
                ssim /= 5.0
                if cur_seen_flag:
                    seen_psnr.append(psnr)
                    seen_ssim.append(ssim)
                else:
                    unseen_psnr.append(psnr)
                    unseen_ssim.append(ssim)
            elif mode == 'pose':
                if cur_seen_flag:
                    seen_psnr.append(0.0)
                    seen_ssim.append(0.0)
                else:
                    unseen_psnr.append(0.0)
                    unseen_ssim.append(0.0)

            if mode == 'all':
                cur_seen_flag = seen_flag[0].item() > 0
                if cur_seen_flag:
                    seen_rot.append(float('inf'))
                    seen_trans.append(float('inf'))
                else:
                    unseen_rot.append(float('inf'))
                    unseen_trans.append(float('inf'))
            elif mode == 'pose' or mode == 'joint':
                rot, trans = 0.0, 0.0
                if not config.train.use_gt_pose:
                    for seq_idx in range(4):
                        cur_rot, cur_trans = eval_utils.compute_pose_metric(camPose_return['pred'][seq_idx].detach().cpu(), 
                                                                            camPose_return['gt'][seq_idx].detach().cpu())
                        rot += cur_rot
                        trans += cur_trans
                rot /= 5.0
                trans /= 5.0
                if cur_seen_flag:
                    seen_rot.append(rot)
                    seen_trans.append(trans)
                else:
                    unseen_rot.append(rot)
                    unseen_trans.append(trans)

            # visualize reconstruction
            if batch_idx % config.eval_vis_freq == 0 and mode != 'pose' and rank == 0:
                vis_utils.vis_seq_sv_mv(vid_clips=sample['images'][:,:5],
                                vid_masks=sample['fg_probabilities'][:,:5],
                                recon_clips=rendered_imgs,
                                recon_masks=rendered_masks,
                                iter_num=str(batch_idx),
                                output_dir=output_dir,
                                subfolder='test_seq',
                                inv_normalize=config.train.normalize_img)
                vis_utils.vis_seq(vid_clips=sample['images'][:,5:],
                                vid_masks=sample['fg_probabilities'][:,5:],
                                recon_clips=rendered_imgs_nvs,
                                recon_masks=rendered_masks_nvs,
                                iter_num=str(batch_idx)+'_nvs_',
                                output_dir=output_dir,
                                subfolder='test_seq')


                # 360-degree NVS
                for idx in range(b):
                    rendered_imgs_results, rendered_masks_results = [], []
                    all_feature = features_mv[idx].unsqueeze(0).repeat(7,1,1,1,1)   # [N,C,D,H,W]
                    all_density = densities_mv[idx].unsqueeze(0).repeat(7,1,1,1,1).clamp(max=1.0)   # [N,1,D,H,W]
                    for pose_idx in range(4):
                        cameras = {
                            'K': sample['K_cv2'][idx][0:1].repeat(7,1,1),                      # [N,3,3]
                            'R': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,:3],
                            'T': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,3],
                        }
                        rendered_imgs, rendered_masks = model.module.render(cameras, all_feature, all_density)           # [N,c,h,w], [N,1,h,w]
                        rendered_imgs_results.append(rendered_imgs.detach())
                        rendered_masks_results.append(rendered_masks.detach())
                    rendered_imgs_results = torch.cat(rendered_imgs_results, dim=0)
                    rendered_masks_results = torch.cat(rendered_masks_results, dim=0)
                    rendered_imgs_results = rendered_imgs_results.clip(min=0.0, max=1.0)
                    vis_utils.vis_NVS(imgs=rendered_imgs_results,
                                        masks=rendered_masks_results, 
                                        img_name=str(batch_idx) + '_' + str(idx),
                                        output_dir=output_dir,
                                        subfolder='test_seq',
                                        inv_normalize=config.train.normalize_img)

    unseen_psnr = np.array(unseen_psnr).mean()
    unseen_ssim = np.array(unseen_ssim).mean()
    unseen_rot = np.array(unseen_rot).mean()
    unseen_trans = np.array(unseen_trans).mean()
    seen_psnr = np.array(seen_psnr).mean()
    seen_ssim = np.array(seen_ssim).mean()
    seen_rot = np.array(seen_rot).mean()
    seen_trans = np.array(seen_trans).mean()
    
    print('unseen: PSNR {}, ssim {}'.format(unseen_psnr, unseen_ssim))
    print('unseen: Rot {}, Trans {}'.format(unseen_rot, unseen_trans))
    print('seen: PSNR {}, ssim {}'.format(seen_psnr, seen_ssim))
    print('seen: Rot {}, Trans {}'.format(seen_rot, seen_trans))

    psnr = 0.5 * (unseen_psnr + seen_psnr)
    rot = 0.5 * (unseen_rot + seen_rot)

    return_dict = {
        'unseen_psnr': unseen_psnr,
        'unseen_ssim': unseen_ssim,
        'unseen_rot': unseen_rot,
        'unseen_trans': unseen_trans,
        'seen_psnr': seen_psnr,
        'seen_ssim': seen_ssim,
        'seen_rot': seen_rot,
        'seen_trans': seen_trans,
    }
    return psnr, rot, return_dict


def validate(config, loader, dataset, model, epoch, output_dir, device, rank):
    '''
    validation for only using both 2D and 3D pose estimator
    '''
    model.eval()
    mode = config.train.parameter
    assert mode in ['pose_head', 'pose', 'joint']
 
    unseen_psnr = []
    unseen_ssim = []
    seen_psnr = []
    seen_ssim = []
    unseen_rot = []
    unseen_trans = []
    seen_rot = []
    seen_trans = []

    # sample NVS camera poses
    num_views_all = 4 * 7
    if not config.train.canonicalize:
        elev = torch.linspace(0, 360, num_views_all)
        azim = torch.linspace(0, 0, num_views_all) + 180
    else:
        elev = torch.linspace(0, 0, num_views_all)
        azim = torch.linspace(0, 360, num_views_all) + 180
    NVS_R_all, NVS_T_all = look_at_view_transform(dist=config.render.camera_z, elev=elev, azim=azim)  # [N,3,3], [N,3]
    NVS_pose_all = torch.cat([NVS_R_all, NVS_T_all.view(-1,3,1)], dim=-1)  # [N,3,4]

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            if batch_idx % config.eval_vis_freq != 0:
                continue

            clips = sample['images'].to(device)
            clips_nvs = clips[:,5:]
            clips = clips[:,:5]
            masks = sample['fg_probabilities'].to(device)
            masks = masks[:,:5]
            K_cv2 = sample['K_cv2'].to(device)
            K_cv2 = K_cv2[:,:5]
            b, t, c, h, w = clips.shape
            seen_flag = sample['seen_flag']
            cur_seen_flag = seen_flag[0].item() > 0
            
            clips = clips.reshape(b*t,c,h,w)
            features_raw = model.module.encoder_3d.get_feat3D(clips)               # [b*t,C,D,H,W]
            _, C, D, H, W = features_raw.shape
            clips = clips.reshape(b,t,c,h,w)
            features_raw = features_raw.reshape(b,t,C,D,H,W)
            
            if not config.train.use_gt_pose:
                # predict relative poses
                pose_feat_3d = model.module.encoder_traj(features_raw, return_features=True)    # [b*(t-1),1024]
                pose_feat_2d = model.module.encoder_traj_2d(clips, return_features=True)        # [b*(t-1),1024]
                pose_feat = torch.cat([pose_feat_3d, pose_feat_2d], dim=-1)                     # [b*(t-1),2048]
                pred = model.module.pose_head(pose_feat)                                        # [b*(t-1), 8]
                poses_cam, conf = pred.split([model.module.encoder_traj.pose_dim, 1], dim=-1)
                tmp = torch.zeros_like(poses_cam)
                tmp[:,:4] = F.normalize(poses_cam[:,:4])
                tmp[:,4:] = poses_cam[:,4:]
                poses_cam = tmp
                camPoseRel_cv2 = model.module.encoder_traj.toSE3(poses_cam)                                               # [b*(t-1),4,4], relative cam pose in cv2 frame
                
                # get camera extrinsics and pose
                canonical_pose_cv2 = dataset.get_canonical_pose_cv2(device=device)                                  # [4,4]
                canonical_extrinsics_cv2 = dataset.get_canonical_extrinsics_cv2(device=device)
                camPoses_cv2 = canonical_pose_cv2.unsqueeze(0) @ camPoseRel_cv2
                camE_cv2 = torch.inverse(camPoses_cv2)                                                              # [b*(t-1),4,4], canonicalized extrinsics
                camE_cv2 = camE_cv2.reshape(b,t-1,4,4)
                camPoses_cv2 = camPoses_cv2.reshape(b,t-1,4,4)
                camPoses_cv2 = torch.cat([canonical_pose_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camPoses_cv2], dim=1)
                camE_cv2 = torch.cat([canonical_extrinsics_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camE_cv2], dim=1)  # [b,t,4,4]
                poses_cam_gt = sample['cam_poses_rel_cv2'][:,1:5].to(device).reshape(b*(t-1),4,4)
                poses_cam_gt = geo_utils.mat2quat(poses_cam_gt)
                camPose_return = {'gt': poses_cam_gt, 'pred': poses_cam}
            else:
                # use ground truth poses
                camE_cv2 = sample['cam_extrinsics_cv2_canonicalized'][:,:5].to(device)     # [b,t,4,4]
                camPoses_cv2 = sample['cam_poses_cv2_canonicalized'][:,:5].to(device)
                poses_cam_gt = sample['cam_poses_rel_cv2'][:,1:5].to(device).reshape(b*(t-1),4,4)
                poses_cam_gt = geo_utils.mat2quat(poses_cam_gt)
                camPose_return = {'gt': poses_cam_gt, 'pred': poses_cam_gt}
            
            # parse camera for rendering using first five views
            camE_cv2 = camE_cv2.repeat(1,2,1,1)             # [b,2*t,4,4]
            camPoses_cv2 = camPoses_cv2.repeat(1,2,1,1)     # [b,2*t,4,4]
            camK = sample['K_cv2'][:,:5].repeat(1,2,1,1)          # [b,2*t,3,3]
            cameras = {
                'R': camE_cv2.reshape(b*2*t,4,4)[:,:3,:3],                # [b*2t,3,3]
                'T': camE_cv2.reshape(b*2*t,4,4)[:,:3,3],                 # [b*2t,3]
                'K': camK.reshape(b*2*t,3,3).to(device)                   # [b*2t,3,3]
            }

            if 'pose' not in mode:
                # transform features using camera poses
                features_transformed = model.module.rotate(voxels=features_raw, camPoses_cv2=camPoses_cv2[:,:t], grid_size=D)   # [b,t,C=128,D=16,H,W]

                # get prediction using features of some views
                # 1) features of first 3 views and render using last two cams
                # 2) features of last 2 views and render using first 3 cams
                features_3v = model.module.encoder_3d.fuse(features_transformed[:,:3])    # [b,C=128,D=16,H,W]
                features_2v = model.module.encoder_3d.fuse(features_transformed[:,-2:])
                densities_3v2v = model.module.encoder_3d.get_density3D(torch.cat([features_3v, features_2v], dim=0))    # [2*b,C=1,D=32,H,W]
                features_3v2v = model.module.encoder_3d.get_render_features(torch.cat([features_3v, features_2v], dim=0))  # [2*b,C=16,D=32,H,W]
                features_3v = features_3v2v[:b].unsqueeze(1).repeat(1,2,1,1,1,1)    # [b,2,C=16,D=32,H,W]
                features_2v = features_3v2v[b:].unsqueeze(1).repeat(1,3,1,1,1,1)    # [b,3,C=16,D=32,H,W]
                features_3v2v = torch.cat([features_2v, features_3v], dim=1)    # [b,t,C=16,D=32,H,W]
                densities_3v = densities_3v2v[:b].unsqueeze(1).repeat(1,2,1,1,1,1)  # [b,2,1,D,H,W]
                densities_2v = densities_3v2v[b:].unsqueeze(1).repeat(1,3,1,1,1,1)
                densities_3v2v = torch.cat([densities_2v, densities_3v], dim=1) # [b,t,1,D,H,W]
                
                # get prediction using multi-view information
                cur_time = time.time()
                features_mv = model.module.encoder_3d.fuse(features_transformed)  # [b,t,C=128,D=16,H,W] -> [b,C,D,H,W]
                densities_mv = model.module.encoder_3d.get_density3D(features_mv)  # [b,1,D=32,H,W]
                features_mv = model.module.encoder_3d.get_render_features(features_mv)    # [b,C=16,D=32,H,W]
                _, C2, D2, H2, W2 = features_mv.shape
                features_all = features_mv.unsqueeze(1).repeat(1,t,1,1,1,1)
                features_all = torch.cat([features_3v2v, features_all], dim=1).reshape(b*2*t,C2,D2,H2,W2)    # [b,2*t,C,D,H,W] -> [b*2*t,C,D,H,W]
                densities_all = densities_mv.unsqueeze(1).repeat(1,t,1,1,1,1)
                densities_all = torch.cat([densities_3v2v, densities_all], dim=1).reshape(b*2*t,1,D2,H2,W2)

                rendered_imgs, rendered_masks, origin_proj = model.module.render(cameras, features_all, densities_all, return_origin_proj=True)
                rendered_imgs = rendered_imgs.reshape(b,2*t,c,h,w)
                rendered_masks = rendered_masks.reshape(b,2*t,1,h,w)

                # parse cameras for rendering using novel views
                cameras_nvs = {
                    'R': sample['cam_extrinsics_cv2_canonicalized'][:,5:].reshape(b*5,4,4)[:,:3,:3].to(device),                # [b*t,3,3]
                    'T': sample['cam_extrinsics_cv2_canonicalized'][:,5:].reshape(b*5,4,4)[:,:3,3].to(device),                 # [b*t,3]
                    'K': sample['K_cv2'][:,5:].reshape(b*5,3,3).to(device)                   # [b*t,3,3]
                }
                features_all = features_mv.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t,C2,D2,H2,W2)
                densities_all = densities_mv.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t,1,D2,H2,W2)
                rendered_imgs_nvs, rendered_masks_nvs = model.module.render(cameras_nvs, features_all, densities_all, return_origin_proj=False)
                rendered_imgs_nvs = rendered_imgs_nvs.reshape(b,5,c,h,w)
                rendered_masks_nvs = rendered_masks_nvs.reshape(b,5,1,h,w)

                # evaluate
                rendered_imgs = rendered_imgs.clip(min=0.0, max=1.0)
                rendered_imgs_nvs = rendered_imgs_nvs.clip(min=0.0, max=1.0)

                psnr, ssim = 0.0, 0.0
                for seq_idx in range(5):
                    cur_img_recon = rendered_imgs_nvs[0, seq_idx].permute(1,2,0).detach().cpu().numpy()
                    cur_img_gt = clips_nvs[0, seq_idx].permute(1,2,0).cpu().numpy()
                    cur_psnr, cur_ssim = eval_utils.compute_img_metric(cur_img_recon, cur_img_gt)
                    psnr += cur_psnr
                    ssim += cur_ssim
                psnr /= 5.0
                ssim /= 5.0
                if cur_seen_flag:
                    seen_psnr.append(psnr)
                    seen_ssim.append(ssim)
                else:
                    unseen_psnr.append(psnr)
                    unseen_ssim.append(ssim)
            else:
                if cur_seen_flag:
                    seen_psnr.append(0.0)
                    seen_ssim.append(0.0)
                else:
                    unseen_psnr.append(0.0)
                    unseen_ssim.append(0.0)

            rot, trans = 0.0, 0.0
            if not config.train.use_gt_pose:
                for seq_idx in range(4):
                    cur_rot, cur_trans = eval_utils.compute_pose_metric(camPose_return['pred'][seq_idx].detach().cpu(), 
                                                                        camPose_return['gt'][seq_idx].detach().cpu())
                    rot += cur_rot
                    trans += cur_trans
            rot /= 5.0
            trans /= 5.0
            if cur_seen_flag:
                seen_rot.append(rot)
                seen_trans.append(trans)
            else:
                unseen_rot.append(rot)
                unseen_trans.append(trans)

            # visualize reconstruction
            if batch_idx % config.eval_vis_freq == 0 and (mode != 'pose' and mode != 'pose_head') and rank == 0:
                vis_utils.vis_seq_sv_mv(vid_clips=sample['images'][:,:5],
                                vid_masks=sample['fg_probabilities'][:,:5],
                                recon_clips=rendered_imgs,
                                recon_masks=rendered_masks,
                                iter_num=str(batch_idx),
                                output_dir=output_dir,
                                subfolder='test_seq',
                                inv_normalize=config.train.normalize_img)
                vis_utils.vis_seq(vid_clips=sample['images'][:,5:],
                                vid_masks=sample['fg_probabilities'][:,5:],
                                recon_clips=rendered_imgs_nvs,
                                recon_masks=rendered_masks_nvs,
                                iter_num=str(batch_idx)+'_nvs_',
                                output_dir=output_dir,
                                subfolder='test_seq')


                # 360-degree NVS
                for idx in range(b):
                    rendered_imgs_results, rendered_masks_results = [], []
                    all_feature = features_mv[idx].unsqueeze(0).repeat(7,1,1,1,1)   # [N,C,D,H,W]
                    all_density = densities_mv[idx].unsqueeze(0).repeat(7,1,1,1,1).clamp(max=1.0)   # [N,1,D,H,W]
                    for pose_idx in range(4):
                        cameras = {
                            'K': sample['K_cv2'][idx][0:1].repeat(7,1,1),                      # [N,3,3]
                            'R': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,:3],
                            'T': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,3],
                        }
                        rendered_imgs, rendered_masks = model.module.render(cameras, all_feature, all_density)           # [N,c,h,w], [N,1,h,w]
                        rendered_imgs_results.append(rendered_imgs.detach())
                        rendered_masks_results.append(rendered_masks.detach())
                    rendered_imgs_results = torch.cat(rendered_imgs_results, dim=0)
                    rendered_masks_results = torch.cat(rendered_masks_results, dim=0)
                    rendered_imgs_results = rendered_imgs_results.clip(min=0.0, max=1.0)
                    vis_utils.vis_NVS(imgs=rendered_imgs_results,
                                        masks=rendered_masks_results, 
                                        img_name=str(batch_idx) + '_' + str(idx),
                                        output_dir=output_dir,
                                        subfolder='test_seq',
                                        inv_normalize=config.train.normalize_img)

    unseen_psnr = np.array(unseen_psnr).mean()
    unseen_ssim = np.array(unseen_ssim).mean()
    unseen_rot = np.array(unseen_rot).mean()
    unseen_trans = np.array(unseen_trans).mean()
    seen_psnr = np.array(seen_psnr).mean()
    seen_ssim = np.array(seen_ssim).mean()
    seen_rot = np.array(seen_rot).mean()
    seen_trans = np.array(seen_trans).mean()
    
    print('unseen: PSNR {}, ssim {}'.format(unseen_psnr, unseen_ssim))
    print('unseen: Rot {}, Trans {}'.format(unseen_rot, unseen_trans))
    print('seen: PSNR {}, ssim {}'.format(seen_psnr, seen_ssim))
    print('seen: Rot {}, Trans {}'.format(seen_rot, seen_trans))

    psnr = 0.5 * (unseen_psnr + seen_psnr)
    rot = 0.5 * (unseen_rot + seen_rot)

    return_dict = {
        'unseen_psnr': unseen_psnr,
        'unseen_ssim': unseen_ssim,
        'unseen_rot': unseen_rot,
        'unseen_trans': unseen_trans,
        'seen_psnr': seen_psnr,
        'seen_ssim': seen_ssim,
        'seen_rot': seen_rot,
        'seen_trans': seen_trans,
    }
    return psnr, rot, return_dict