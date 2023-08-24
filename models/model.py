import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import torchvision
from utils import train_utils, geo_utils
import random

from models.encoder import Encoder3D
from models.volume_render import VolRender
from models.pose_estimator_3d import PoseEstimator3D
from models.pose_estimator_2d import PoseEstimator2D
from models.rotate import Rotate_world

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


class FORGE(nn.Module):
    def __init__(self, config):
        super(FORGE, self).__init__()

        self.config = config

        self.encoder_3d = Encoder3D(config)
        self.render = VolRender(config)
        self.rotate = Rotate_world(config)
        
        self.encoder_traj = PoseEstimator3D(config)
        self.encoder_traj_2d = PoseEstimator2D()

        dropout_p = 0.5
        self.pose_head = nn.Sequential(*[
            nn.Dropout(p=dropout_p),
            nn.Linear(2048, 512),
            #nn.BatchNorm1d(512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, self.encoder_traj.pose_dim + 1)
        ])
        
    
    def forward(self, sample, dataset, device):
        '''
        Render object using:
            1) Features from some of the views and other cameras
            2) Features from all views
        '''
        b,t_all,_,_,_ = sample['images'].shape

        clips = sample['images'][:,:5].to(device)
        b,t,c,h,w = clips.shape

        # get 3D volume features of all frames
        clips = clips.reshape(b*t,c,h,w)
        features_raw = self.encoder_3d.get_feat3D(clips)               # [b*t,C,D,H,W]
        _, C, D, H, W = features_raw.shape
        clips = clips.reshape(b,t,c,h,w)
        features_raw = features_raw.reshape(b,t,C,D,H,W)
        
        if not self.config.train.use_gt_pose:
            # predict relative poses
            pose_feat_3d = self.encoder_traj(features_raw, return_features=True)    # [b*(t-1),1024]
            pose_feat_2d = self.encoder_traj_2d(clips, return_features=True)        # [b*(t-1),1024]
            pose_feat = torch.cat([pose_feat_3d, pose_feat_2d], dim=-1)             # [b*(t-1),2048]
            pred = self.pose_head(pose_feat)                                        # [b*(t-1), 8]
            poses_cam, conf = pred.split([self.encoder_traj.pose_dim, 1], dim=-1)
            tmp = torch.zeros_like(poses_cam)
            tmp[:,:4] = F.normalize(poses_cam[:,:4])
            tmp[:,4:] = poses_cam[:,4:]
            poses_cam = tmp
            camPoseRel_cv2 = self.encoder_traj.toSE3(poses_cam)                                               # [b*(t-1),4,4], relative cam pose in cv2 frame
            
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
            camPose_return = {'gt': poses_cam_gt, 'pred': poses_cam, 'conf': conf}
            idxs = sequence_from_distance(camPoses_cv2[:,:,:3,3])
        else:
            camPoseRel_cv2 = None
            # use ground truth poses
            if self.config.train.canonicalize:
                camE_cv2 = sample['cam_extrinsics_cv2_canonicalized'].to(device)                                # [b,t,4,4]
                camPoses_cv2 = sample['cam_poses_cv2_canonicalized'].to(device)                                 # [b,t,4,4], relative to first frame
            else:
                camE_cv2 = sample['cam_extrinsics_cv2'].to(device)                                              # [b,t,4,4]
                camPoses_cv2 = sample['cam_poses_cv2'].to(device)
            camPose_return = None
        
        if self.config.train.parameter == 'pose' or self.config.train.parameter == 'pose_head':
            # only train pose estimator, only return pose estimation results to enable larger batch size
            camK = sample['K_cv2'].to(device)[:,:5]
            camera_params = {
                'R': camE_cv2.reshape(b*t,4,4)[:,:3,:3],                # [b*t,3,3]
                'T': camE_cv2.reshape(b*t,4,4)[:,:3,3],                 # [b*t,3]
                'K': camK.reshape(b*t,3,3).to(device)                   # [b*t,3,3]
            }
            camera_params['K'] /= 2.0
            camera_params['K'][:,-1,-1] = 1.0
            cameras = cameras_from_opencv_projection(R=camera_params['R'],
                                                    tvec=camera_params['T'], 
                                                    camera_matrix=camera_params['K'],
                                                    image_size=torch.tensor([self.config.dataset.img_size//2]*2).unsqueeze(0).repeat(b*t,1)).to(device)
            origin = torch.zeros(1,3).to(device)
            origin_proj = cameras.transform_points_screen(origin, eps=1e-6).squeeze()[...,:2]
            return camPose_return, 2 * origin_proj / self.config.dataset.img_size
        
        # parse camera for rendering
        camE_cv2 = torch.cat([camE_cv2, sample['cam_extrinsics_cv2_canonicalized'][:,5:]], dim=1).to(device)
        camPoses_cv2 = torch.cat([camPoses_cv2, sample['cam_poses_cv2_canonicalized'][:,5:]],dim=1).to(device)
        camK = sample['K_cv2'].to(device)
        
        cameras = {
            'R': camE_cv2.reshape(b*t_all,4,4)[:,:3,:3],                # [b*2t,3,3]
            'T': camE_cv2.reshape(b*t_all,4,4)[:,:3,3],                 # [b*2t,3]
            'K': camK.reshape(b*t_all,3,3)                              # [b*2t,3,3]
        }

        # transform features using camera poses
        features_transformed = self.rotate(voxels=features_raw, camPoses_cv2=camPoses_cv2[:,:t], grid_size=D)   # [b,t,C=128,D=16,H,W]
        features_transformed = chose_selected(features_transformed, idxs)

        # get prediction using multi-view information
        features_mv = self.encoder_3d.fuse(features_transformed)            # [b,t,C=128,D=16,H,W] -> [b,C,D,H,W]
        densities_mv = self.encoder_3d.get_density3D(features_mv)           # [b,1,D=32,H,W]
        features_mv = self.encoder_3d.get_render_features(features_mv)      # [b,C=16,D=32,H,W]

        # render
        _, C2, D2, H2, W2 = features_mv.shape
        features_all = features_mv.unsqueeze(1).repeat(1,t_all,1,1,1,1).reshape(b*t_all,C2,D2,H2,W2)    # [b,2*t,C,D,H,W] -> [b*2*t,C,D,H,W]
        densities_all = densities_mv.unsqueeze(1).repeat(1,t_all,1,1,1,1).reshape(b*t_all,1,D2,H2,W2)
        if self.config.dataset.name == 'omniobject3d':
            densities_all = densities_all.clamp(min=0.0, max=1.0)

        rendered_imgs, rendered_masks, origin_proj = self.render(cameras, features_all, densities_all, return_origin_proj=True)
        
        if self.config.train.use_gt_pose:
            return rendered_imgs, rendered_masks
        else:
            return rendered_imgs, rendered_masks, 2 * origin_proj / self.config.dataset.img_size, camPose_return
        


def sequence_from_distance(trans):
    # translations in [b,t,3] shape
    b, t, _ = trans.shape
    trans_canonical = trans[:,0:1,:]    # [b,1,3]
    dist = ((trans - trans_canonical) ** 2).sum(dim=-1)   # [b,t]
    _, idxs = torch.sort(dist, descending=False)   # small to large
    return idxs


def chose_selected(tensor, idxs):
    assert tensor.shape[0] == len(idxs)
    tensor_selected = []
    for i in range(len(idxs)):
        cur_selected = tensor[i][idxs[i]]
        tensor_selected.append(cur_selected)
    tensor_selected = torch.stack(tensor_selected)
    return tensor_selected