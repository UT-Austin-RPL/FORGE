import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F
import itertools

import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist

import argparse
from config.config import config, update_config

from utils import exp_utils, train_utils, eval_utils, vis_utils, geo_utils

from models.model import FORGE as ReconModel
from models.model import sequence_from_distance, chose_selected
from models.perceptual_loss import VGGPerceptualLoss as PerceptualLoss
from dataset.kubric import Kubric
import time
from pytorch3d.renderer import look_at_view_transform
import json
from PIL import Image


def run_demo(args, config, dataset, data, model, model_gt, output_dir, device):
    # data in [b,t,c,h,w]
    model.eval()
    model_gt.eval()
    _, t, c, h, w = data.shape
    b = 1

    for batch_idx, cur_data in enumerate(data):
        clips = cur_data.float().unsqueeze(0).to(device)    # [b=1,t,c,h,w]
        K_cv2 = torch.tensor([[1.38888, 0.0, 0.5], [0.0, 1.38888, 0.5], [0.0, 0.0, 0.0]]) * 256.0
        K_cv2[-1,-1] = 1.0
        K_cv2 = K_cv2.unsqueeze(0).float()

        # predict 3D feature volumes
        clips = clips.reshape(b*t,c,h,w)
        features_raw = model.module.encoder_3d.get_feat3D(clips)                        # [b*t,C,D,H,W]
        _, C, D, H, W = features_raw.shape
        clips = clips.reshape(b,t,c,h,w)
        features_raw = features_raw.reshape(b,t,C,D,H,W)

        # predict camera relative pose
        pose_feat_3d = model.module.encoder_traj(features_raw, return_features=True)    # [b*(t-1),1024]
        pose_feat_2d = model.module.encoder_traj_2d(clips, return_features=True)        # [b*(t-1),1024]
        pose_feat = torch.cat([pose_feat_3d, pose_feat_2d], dim=-1)                     # [b*(t-1),2048]
        pred = model.module.pose_head(pose_feat)                                        # [b*(t-1), 8]
        poses_cam, _ = pred.split([model.module.encoder_traj.pose_dim, 1], dim=-1)
        tmp = torch.zeros_like(poses_cam)
        tmp[:,:4] = F.normalize(poses_cam[:,:4])
        tmp[:,4:] = poses_cam[:,4:]
        poses_cam = tmp

        # optimize pose
        poses_cam = refine_pose(model, dataset, clips, poses_cam.detach(), features_raw.detach(), K_cv2.unsqueeze(0), device)
        
        # get camera extrinsics and pose   
        camPoseRel_cv2 = model.module.encoder_traj.toSE3(poses_cam)
        canonical_pose_cv2 = dataset.get_canonical_pose_cv2(device=device)                                  # [4,4]
        canonical_extrinsics_cv2 = dataset.get_canonical_extrinsics_cv2(device=device)
        camPoses_cv2 = canonical_pose_cv2.unsqueeze(0) @ camPoseRel_cv2
        camE_cv2 = torch.inverse(camPoses_cv2)                                                              # [b*(t-1),4,4], canonicalized extrinsics
        camE_cv2 = camE_cv2.reshape(b,t-1,4,4)
        camPoses_cv2 = camPoses_cv2.reshape(b,t-1,4,4)
        camPoses_cv2 = torch.cat([canonical_pose_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camPoses_cv2], dim=1)# [b,t,4,4]
        camE_cv2 = torch.cat([canonical_extrinsics_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camE_cv2], dim=1)  # [b,t,4,4]
        
        # feature fusion and predict neural volume
        features_transformed = model_gt.module.rotate(voxels=features_raw, camPoses_cv2=camPoses_cv2[:,:t], grid_size=D)   # [b,t,C=128,D=16,H,W]
        idxs = sequence_from_distance(camPoses_cv2[:,:,:3,3])
        features_transformed = chose_selected(features_transformed, idxs)
        features_mv = model_gt.module.encoder_3d.fuse(features_transformed)         # [b,t,C,D,H,W] -> [b,C,D,H,W]
        densities_mv = model_gt.module.encoder_3d.get_density3D(features_mv)        # [b,1,D,H,W]
        features_mv = model_gt.module.encoder_3d.get_render_features(features_mv)   # [b,C,D,H,W]
        
        # sample NVS camera poses
        num_views_all = 4 * 7
        elev = torch.linspace(0, 0, num_views_all)
        azim = torch.linspace(0, 360, num_views_all) + 180
        NVS_R_all, NVS_T_all = look_at_view_transform(dist=config.render.camera_z, elev=elev, azim=azim)
        NVS_pose_all = torch.cat([NVS_R_all, NVS_T_all.view(-1,3,1)], dim=-1)

        # 360-degree NVS
        for idx in range(b):
            rendered_imgs_results, rendered_masks_results = [], []
            all_feature = features_mv[idx].unsqueeze(0).repeat(7,1,1,1,1)   # [N,C,D,H,W]
            all_density = densities_mv[idx].unsqueeze(0).repeat(7,1,1,1,1).clamp(max=1.0)   # [N,1,D,H,W]
            for pose_idx in range(4):
                cameras = {
                    'K': K_cv2.repeat(7,1,1),  # [N,3,3]
                    'R': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,:3],
                    'T': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,3],
                }
                rendered_imgs, rendered_masks = model_gt.module.render(cameras, all_feature, all_density)           # [N,c,h,w], [N,1,h,w]
                rendered_imgs_results.append(rendered_imgs.detach())
                rendered_masks_results.append(rendered_masks.detach())
            rendered_imgs_results = torch.cat(rendered_imgs_results, dim=0)
            rendered_masks_results = torch.cat(rendered_masks_results, dim=0)
            
            print('Saving results to {}'.format(output_dir))
            vis_utils.vis_NVS(imgs=rendered_imgs_results,
                                masks=rendered_masks_results, 
                                img_name=str(batch_idx) + '_' + str(idx),
                                output_dir=output_dir,
                                subfolder='vis_360')
            

def refine_pose(model, dataset, clips, poses_cam, features, K, device):
    b, t, c, h, w = clips.shape
    _, _, C, D, H, W = features.shape

    masks = (clips[:, :, 0:1] > 0.05).float()
    target_imgs = clips.view(b*t,c,h,w)
    target_masks = masks.view(b*t,1,h,w)

    # optimize poses
    poses_cam = poses_cam.detach()
    poses_cam.requires_grad = True
    poses_cam_rot = poses_cam[:,:4].detach()
    poses_cam_trans = poses_cam[:,4:].detach()
    poses_cam_rot.requires_grad = True
    poses_cam_trans.requires_grad = True
    #optimizer = torch.optim.SGD([poses_cam], lr=0.0005, momentum=0.9)
    optimizer = torch.optim.Adam([{'params': poses_cam_rot, 'lr': 0.001},
                                 {'params': poses_cam_trans, 'lr': 0.0005}
                                ], lr=0.001)#, momentum=0.9)

    for iter_idx in range(2000+1):     # 500 iterations should be enough
        poses_cam_normalized = torch.zeros_like(poses_cam)
        poses_cam_normalized[:,:4] = F.normalize(poses_cam_rot)
        poses_cam_normalized[:,4:] = poses_cam_trans
        camPoseRel_cv2 = model.module.encoder_traj.toSE3(poses_cam_normalized)

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
        camK = K.repeat(1,t,1,1)          # [b,2*t,3,3]
        cameras = {
                'R': camE_cv2.reshape(b*1*t,4,4)[:,:3,:3].to(device),                # [b*t,3,3]
                'T': camE_cv2.reshape(b*1*t,4,4)[:,:3,3].to(device),                 # [b*t,3]
                'K': camK.reshape(b*1*t,3,3).to(device)                   # [b*t,3,3]
            }
        
        features_all = features_mv.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t,C2,D2,H2,W2)    # [b,2*t,C,D,H,W] -> [b*2*t,C,D,H,W]
        densities_all = densities_mv.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t,1,D2,H2,W2)
        rendered_imgs, rendered_masks = model.module.render(cameras, features_all, densities_all, return_origin_proj=False)

        # calculate loss
        loss_recon_img = config.loss.recon_rgb * F.mse_loss(rendered_imgs, target_imgs)
        loss_recon_mask = config.loss.recon_mask * F.mse_loss(rendered_masks, target_masks)
        loss_recon = loss_recon_img + loss_recon_mask
        
        # optimize pose
        optimizer.zero_grad()
        loss_recon.backward()
        optimizer.step()
    
    poses_final = torch.zeros_like(poses_cam)
    poses_final[:,:4] = F.normalize(poses_cam_rot)
    poses_final[:,4:] = poses_cam_trans
    return poses_final


def parse_args():
    parser = argparse.ArgumentParser(description='FORGE demo')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    # Get args and config
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
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
    cpt_root = './output/kubric/joint_pose_2d3d/pred_pose_2d3d_joint'
    cpt_name = 'cpt_best_psnr_26.340881009038913_7.545314707482719.pth.tar'
    cpt = torch.load(os.path.join(cpt_root, cpt_name))['state_dict']
    model.load_state_dict(cpt, strict=True)
    model = torch.nn.DataParallel(model)
    
    # use fusion module without joint training
    model_gt = ReconModel(config).to(device)
    cpt_root = './output/kubric/gt_pose/gt_pose'
    cpt_name = 'cpt_best_psnr_31.842686198427398.pth.tar'
    cpt = torch.load(os.path.join(cpt_root, cpt_name))['state_dict']
    del cpt['encoder_traj.out.3.weight']
    del cpt['encoder_traj.out.3.bias']
    model_gt.load_state_dict(cpt, strict=False)
    model_gt = torch.nn.DataParallel(model_gt)

    all_data = []
    for i in range(3):  # case id
        cur_data = []
        for j in range(3):  # frame id
            img_name = './assets/real_images/{}_{}'.format(i+1, j)
            if os.path.isfile(img_name+'.jpg'):
                with Image.open(img_name+'.jpg') as img_pil:
                    img_np = np.asarray(img_pil)[:,:,:3]
            elif os.path.isfile(img_name+'.png'):
                with Image.open(img_name+'.png') as img_pil:
                    img_np = np.asarray(img_pil)[:,:,:3].copy()
                    mask_np = np.uint8(np.asarray(img_pil)[:,:,3:] > 0).copy()
                    img_np *= mask_np
            rgb = Image.fromarray(img_np[:,:,:3])
            rgb = rgb.resize((256, 256), Image.ANTIALIAS)
            rgb = np.asarray(rgb).transpose((2,0,1)) / 255.0   # [3,H,W]
            rgb = torch.from_numpy(rgb)
            cur_data.append(rgb)
        cur_data = torch.stack(cur_data)    # [t,3,h,w]
        all_data.append(cur_data)
    all_data = torch.stack(all_data)    # [b,t,3,h,w]

    val_data = Kubric(config, split='test')

    run_demo(args, config, dataset=val_data, data=all_data, model=model, model_gt=model_gt, output_dir=output_dir, device=device)


if __name__ == '__main__':
    main()