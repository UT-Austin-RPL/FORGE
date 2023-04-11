import numpy as np
import torch
import skimage.measure
import skimage.metrics
from utils import geo_utils


def compute_img_metric(rgb, gt):
    ssim = skimage.metrics.structural_similarity(gt, rgb, multichannel=True, data_range=1)
    psnr = skimage.metrics.peak_signal_noise_ratio(gt, rgb, data_range=1)
    return psnr, ssim


def compute_pose_metric(pred, gt):
    '''
    https://github.com/kentsommer/tensorflow-posenet/blob/master/test.py
    pred, gt in shape [7]
    '''
    q_pred = pred[:4].numpy()
    q = gt[:4].numpy()
    d = np.abs(np.sum(np.multiply(q_pred, q)))
    theta = 2 * np.arccos(d) * 180 / np.pi

    t_pred = pred[4:].numpy()
    t = gt[4:].numpy()
    t_error = np.linalg.norm(t_pred - t)
    return theta, t_error


def permute_clips(clips, gt_poses, nvs_extr, canonical_id, clips_only=False, return_permutation=False):
    # clips in shape [1,t,c,h,w]
    t = clips.shape[1]
    canonical_id = int(canonical_id)

    # permute clips
    if canonical_id == 0:
        permute = list(range(t))
    elif canonical_id == (t-1):
        permute = [canonical_id] + list(range(t-1))
    else:
        permute = [canonical_id] + list(range(canonical_id)) + list(range(canonical_id+1,t))
    
    clips = clips[:, permute]
    if clips_only:
        return clips
    gt_poses = gt_poses.squeeze()   # relative poses
    nvs_poses = torch.inverse(nvs_extr.squeeze())

    canonical_pose = gt_poses[canonical_id]  # [4,4]
    gt_poses = geo_utils.get_relative_pose(canonical_pose, gt_poses)  # [t,4,4]
    gt_poses = gt_poses[permute].unsqueeze(0)
    
    tmp = torch.inverse(torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, 1.5],
                                                      [0.0, 0.0, 0.0, 1.0]]))
    nvs_poses = geo_utils.get_relative_pose(nvs_poses[canonical_id], nvs_poses)
    nvs_poses = geo_utils.canonicalize_poses(tmp, nvs_poses)
    nvs_extr = torch.inverse(nvs_poses).unsqueeze(0)

    if return_permutation:
        return clips, gt_poses, nvs_extr, permute
    return clips, gt_poses, nvs_extr