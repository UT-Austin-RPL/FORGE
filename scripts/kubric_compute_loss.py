import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import geo_utils, train_utils


def compute_reconstruction_loss(config, epoch, sample, dataset, model, losses, device, perceptual_loss=None):
    '''
    Train model with GT poses
    rendered images in shape [b,2t,c,h,w]
    '''
    rendered_imgs, rendered_masks = model(sample, dataset, device)  # [b*2*t,c,h,w]

    clips = sample['images'].to(device)             # [b,t,c,h,w]
    masks = sample['fg_probabilities'].to(device)
    b, t, c, h, w = clips.shape
    target_imgs = clips.view(b*t,c,h,w)
    target_masks = masks.view(b*t,1,h,w)

    rendered_imgs = rendered_imgs.reshape(b,2*t,c,h,w)
    rendered_masks = rendered_masks.reshape(b,2*t,1,h,w)
    
    loss_recon = 0.0
    loss_recon_img_sv = config.loss.recon_rgb * F.mse_loss(rendered_imgs[:,:t].reshape(-1,c,h,w), target_imgs)
    loss_recon_mask_sv = config.loss.recon_mask * F.mse_loss(rendered_masks[:,:t].reshape(-1,1,h,w), target_masks)
    loss_recon_img_mv = config.loss.recon_rgb * F.mse_loss(rendered_imgs[:,t:].reshape(-1,c,h,w), target_imgs)
    loss_recon_mask_mv = config.loss.recon_mask * F.mse_loss(rendered_masks[:,t:].reshape(-1,1,h,w), target_masks)
    loss_recon += loss_recon_img_sv + loss_recon_mask_sv
    loss_recon += loss_recon_img_mv + loss_recon_mask_mv
    losses['recon_img_sv'] = loss_recon_img_sv.item()
    losses['recon_mask_sv'] = loss_recon_mask_sv.item()
    losses['recon_img_mv'] = loss_recon_img_mv.item()
    losses['recon_mask_mv'] = loss_recon_mask_mv.item()
    
    if config.loss.perceptual_img > 0:
        target_imgs = target_imgs.reshape(b,t,c,h,w).repeat(1,2,1,1,1).reshape(b*2*t,c,h,w)
        loss_perceptual = config.loss.perceptual_img * perceptual_loss(rendered_imgs.reshape(-1,c,h,w), target_imgs).mean()
        losses['perceptual_img'] = loss_perceptual.item()
        loss_recon += loss_perceptual
    return loss_recon, losses, rendered_imgs.reshape(b,2*t,c,h,w), rendered_masks.reshape(b,2*t,1,h,w)


def compute_pose_loss(config, epoch, sample, dataset, model, losses, device, perceptual_loss=None):
    '''
    Train to fuse the features of two pose estimators
    '''
    b, t = sample['images'].shape[:2]
    camPose_return, origin_proj = model(sample, dataset, device)  # [b*t,c,h,w]

    loss_recon = 0.0
    loss_pose = F.mse_loss(camPose_return['pred'][:,:4], camPose_return['gt'][:,:4])
    loss_trans = F.mse_loss(camPose_return['pred'][:,4:], camPose_return['gt'][:,4:])
    losses['pose'] = loss_pose.item()
    loss_recon += loss_pose
    losses['trans'] = loss_trans.item()
    loss_recon += loss_trans
    
    if config.loss.regu_origin_proj > 0 and epoch >= 100:
        loss_regu_origin = config.loss.regu_origin_proj * F.mse_loss(origin_proj, 
                                                                     torch.tensor([0.5, 0.5]).reshape(1,2).to(origin_proj))
        losses['regu_origin'] = loss_regu_origin.item()
        loss_recon += loss_regu_origin

    return loss_recon, losses, None, None


def compute_all_loss(config, epoch, sample, dataset, model, losses, device, perceptual_loss=None):
    '''
    Train model (except encoder 3d backbone) with both 2D reconstruction and pose loss
    ''' 
    rendered_imgs, rendered_masks, origin_proj, camPose_return = model(sample, dataset, device)  # [b*2*t,c,h,w]

    clips = sample['images'].to(device)             # [b,t,c,h,w]
    masks = sample['fg_probabilities'].to(device)
    b, t, c, h, w = clips.shape
    target_imgs = clips.view(b*t,c,h,w)
    target_masks = masks.view(b*t,1,h,w)

    rendered_imgs = rendered_imgs.reshape(b,2*t,c,h,w)
    rendered_masks = rendered_masks.reshape(b,2*t,1,h,w)
    
    # calculate loss for reconstruction
    loss_recon = 0.0
    loss_recon_img_sv = config.loss.recon_rgb * F.mse_loss(rendered_imgs[:,:t].reshape(-1,c,h,w), target_imgs)
    loss_recon_mask_sv = config.loss.recon_mask * F.mse_loss(rendered_masks[:,:t].reshape(-1,1,h,w), target_masks)
    loss_recon_img_mv = config.loss.recon_rgb * F.mse_loss(rendered_imgs[:,t:].reshape(-1,c,h,w), target_imgs)
    loss_recon_mask_mv = config.loss.recon_mask * F.mse_loss(rendered_masks[:,t:].reshape(-1,1,h,w), target_masks)
    loss_recon += loss_recon_img_sv + loss_recon_mask_sv
    loss_recon += loss_recon_img_mv + loss_recon_mask_mv
    losses['recon_img_sv'] = loss_recon_img_sv.item()
    losses['recon_mask_sv'] = loss_recon_mask_sv.item()
    losses['recon_img_mv'] = loss_recon_img_mv.item()
    losses['recon_mask_mv'] = loss_recon_mask_mv.item()

    loss_pose = F.mse_loss(camPose_return['pred'][:,:4], camPose_return['gt'][:,:4])
    losses['pose'] = loss_pose.item()
    loss_recon += loss_pose

    loss_trans = F.mse_loss(camPose_return['pred'][:,4:], camPose_return['gt'][:,4:])
    losses['trans'] = loss_trans.item()
    loss_recon += loss_trans
    
    if config.loss.perceptual_img > 0:
        target_imgs = target_imgs.reshape(b,t,c,h,w).repeat(1,2,1,1,1).reshape(b*2*t,c,h,w)
        loss_perceptual = config.loss.perceptual_img * perceptual_loss(rendered_imgs.reshape(-1,c,h,w), target_imgs).mean()
        losses['perceptual_img'] = loss_perceptual.item()
        loss_recon += loss_perceptual

    if config.loss.regu_origin_proj > 0:
        loss_regu_origin = config.loss.regu_origin_proj * F.mse_loss(origin_proj, 
                                                                     torch.tensor([0.5, 0.5]).reshape(1,2).repeat(b*2*t,1).to(origin_proj))
        losses['regu_origin'] = loss_regu_origin.item()
        loss_recon += loss_regu_origin

    return loss_recon, losses, rendered_imgs.reshape(b,2*t,c,h,w), rendered_masks.reshape(b,2*t,1,h,w)



def compute_all_loss_nvs(config, epoch, sample, dataset, model, losses, device, perceptual_loss=None):
    '''
    Train model with both 2D reconstruction and pose loss
    '''
    rendered_imgs, rendered_masks, origin_proj, camPose_return = model(sample, dataset, device)  # [b*2*t,c,h,w]

    clips = sample['images'][:,:5].to(device)              # [b,t,c,h,w]
    clips_nvs = sample['images'][:,5:].to(device)
    masks = sample['fg_probabilities'][:,:5].to(device)
    masks_nvs = sample['fg_probabilities'][:,5:].to(device)
    b, t, c, h, w = clips.shape
    t_nvs = clips_nvs.shape[1]
    t_all = t + t_nvs

    target_imgs = torch.cat([clips, clips_nvs],dim=1).view(b*t_all,c,h,w)
    target_masks = torch.cat([masks, masks_nvs],dim=1).view(b*t_all,1,h,w)
    rendered_imgs = rendered_imgs.reshape(b,t_all,c,h,w)
    rendered_masks = rendered_masks.reshape(b,t_all,1,h,w)
    
    # calculate loss for reconstruction
    loss_recon = 0.0
    loss_recon_img = config.loss.recon_rgb * F.mse_loss(rendered_imgs[:,:t].reshape(-1,c,h,w), clips.reshape(-1,c,h,w))
    loss_recon_mask = config.loss.recon_mask * F.mse_loss(rendered_masks[:,:t].reshape(-1,1,h,w), masks.reshape(-1,1,h,w))
    loss_recon += loss_recon_img + loss_recon_mask
    losses['recon_img'] = loss_recon_img.item()
    losses['recon_mask'] = loss_recon_mask.item()
    loss_recon_img_nvs = config.loss.recon_rgb * F.mse_loss(rendered_imgs[:,t:].reshape(-1,c,h,w), clips_nvs.reshape(-1,c,h,w))
    loss_recon_mask_nvs = config.loss.recon_mask * F.mse_loss(rendered_masks[:,t:].reshape(-1,1,h,w), masks_nvs.reshape(-1,1,h,w))
    loss_recon += loss_recon_img_nvs + loss_recon_mask_nvs
    losses['recon_img_nvs'] = loss_recon_img_nvs.item()
    losses['recon_mask_nvs'] = loss_recon_mask_nvs.item()

    loss_pose = F.mse_loss(camPose_return['pred'][:,:4], camPose_return['gt'][:,:4])
    losses['pose'] = loss_pose.item()
    loss_recon += loss_pose

    loss_trans = F.mse_loss(camPose_return['pred'][:,4:], camPose_return['gt'][:,4:])
    losses['trans'] = loss_trans.item()
    loss_recon += loss_trans
    
    if config.loss.perceptual_img > 0:
        loss_perceptual = config.loss.perceptual_img * perceptual_loss(rendered_imgs.reshape(-1,c,h,w), target_imgs).mean()
        losses['perceptual_img'] = loss_perceptual.item()
        loss_recon += loss_perceptual

    if config.loss.regu_origin_proj > 0:
        loss_regu_origin = config.loss.regu_origin_proj * F.mse_loss(origin_proj, 
                                                                     torch.tensor([0.5, 0.5]).reshape(1,2).to(origin_proj))
        losses['regu_origin'] = loss_regu_origin.item()
        loss_recon += loss_regu_origin

    return loss_recon, losses, rendered_imgs.reshape(b,t_all,c,h,w), rendered_masks.reshape(b,t_all,1,h,w)
