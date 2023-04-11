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


logger = logging.getLogger(__name__)

adjust_iter_num = [13000, 25000, 35000, 50000]


def compute_pose_loss(sample, model, losses, device):
    clips = sample['images'].to(device)     # [b,t,c,h,w]
    b,t,c,h,w = clips.shape
    poses_gt = sample['cam_poses_rel_cv2'][:,1:5].to(device).reshape(b*(t-1),4,4)
    poses_gt = geo_utils.mat2quat(poses_gt)

    poses_pred = model(clips)    # [b*(t-1), 7]
    tmp = torch.zeros_like(poses_pred)
    tmp[:,:4] = F.normalize(poses_pred[:,:4])
    tmp[:,4:] = poses_pred[:,4:]
    poses_pred = tmp

    loss = 0.0
    loss_pose = F.mse_loss(poses_pred[:,:4], poses_gt[:,:4])
    loss_trans = F.mse_loss(poses_pred[:,4:], poses_gt[:,4:])
    losses['pose'] = loss_pose.item()
    loss += loss_pose
    losses['trans'] = loss_trans.item()
    loss += loss_trans

    return loss, losses


def train_epoch(config, loader, dataset, model, optimizer, epoch, output_dir, device, rank):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()
    
    model.train()

    batch_end = time.time()
    for batch_idx, sample in enumerate(loader):

        # adjust lr
        iter_num = batch_idx + len(loader) * epoch
        if iter_num in adjust_iter_num:
            train_utils.adjust_lr(config, optimizer, iter_num, adjust_iter_num)
        
        #sample = train_utils.dict_to_cuda(sample)
        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        losses = {}

        # reconstruction loss
        loss_pose, losses = compute_pose_loss(sample, model, losses, device)
        time_meters.add_loss_value('Reconstruction time', time.time() - end)
        end = time.time()

        total_loss = loss_pose

        dist.barrier()

        total_loss = total_loss / config.train.accumulation_step
        total_loss.backward()
        if (batch_idx+1) % config.train.accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad()

        for k, v in losses.items():
            if v is not None:
                loss_meters.add_loss_value(k, v)
        
        time_meters.add_loss_value('Batch time', time.time() - batch_end)

        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, recon {recon_time:.3f}s, all {batch_time:.3f}s, Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].val,
                recon_time=time_meters.average_meters['Reconstruction time'].val,
                batch_time=time_meters.average_meters['Batch time'].val
            )
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.6f} ({loss.avg:.6f}), '.format(
                        k, loss=loss_meters.average_meters[k]
                )
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)

        batch_end = time.time()


def validate(config, loader, dataset, model, epoch, output_dir, device, rank):
    model.eval()

    unseen_rot = []
    unseen_trans = []
    seen_rot = []
    seen_trans = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            if batch_idx % config.eval_vis_freq != 0:
                continue

            clips = sample['images'].to(device)
            clips = clips[:,:5]
            b, t, c, h, w = clips.shape
            seen_flag = sample['seen_flag']
            poses_gt = sample['cam_poses_rel_cv2'][:,1:5].to(device).reshape(b*(t-1),4,4)
            poses_gt = geo_utils.mat2quat(poses_gt)     # [b*(t-1),7]

            poses_pred = model(clips)    # [b*(t-1), 7]
            tmp = torch.zeros_like(poses_pred)
            tmp[:,:4] = F.normalize(poses_pred[:,:4])
            tmp[:,4:] = poses_pred[:,4:]
            poses_pred = tmp
        
            rot, trans = 0.0, 0.0
            for seq_idx in range(4):
                cur_rot, cur_trans = eval_utils.compute_pose_metric(poses_pred[seq_idx].detach().cpu(), 
                                                                    poses_gt[seq_idx].detach().cpu())
                rot += cur_rot
                trans += cur_trans

            rot /= 5.0
            trans /= 5.0
            cur_seen_flag = seen_flag[0].item() > 0
            if cur_seen_flag:
                seen_rot.append(rot)
                seen_trans.append(trans)
            else:
                unseen_rot.append(rot)
                unseen_trans.append(trans)

    unseen_rot = np.array(unseen_rot).mean()
    unseen_trans = np.array(unseen_trans).mean()
    seen_rot = np.array(seen_rot).mean()
    seen_trans = np.array(seen_trans).mean()
    
    print('unseen: Rot {}, Trans {}'.format(unseen_rot, unseen_trans))
    print('seen: Rot {}, Trans {}'.format(seen_rot, seen_trans))

    rot = 0.5 * (seen_rot + unseen_rot)
    trans = 0.5 * (seen_trans + unseen_trans)

    return_dict = {
        'unseen_rot': unseen_rot,
        'unseen_trans': unseen_trans,
        'seen_rot': seen_rot,
        'seen_trans': seen_trans,
    }
    return rot, trans, return_dict