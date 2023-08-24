import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import geo_utils

from models.encoder import Encoder3D
from models.volume_render import VolRender
from models.pose_estimator_3d import PoseEstimator3D
from models.rotate import Rotate_world

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


class FORGE_poseEstimator3D(nn.Module):
    def __init__(self, config):
        super(FORGE_poseEstimator3D, self).__init__()

        self.config = config

        self.encoder_3d = Encoder3D(config)
        self.render = VolRender(config)
        self.rotate = Rotate_world(config)
        self.encoder_traj = PoseEstimator3D(config)
        
    
    def forward(self, sample, dataset, device):
        '''
        Render object using:
            1) Features from some of the views and other cameras
            2) Features from all views
        '''
        
        clips = sample['images'].to(device)
        b, t, c, h, w = clips.shape

        # get 3D volume features of all frames
        clips = clips.reshape(b*t,c,h,w)
        features_raw = self.encoder_3d.get_feat3D(clips)    # [b*t,C,D,H,W]
        _, C, D, H, W = features_raw.shape
        clips = clips.reshape(b,t,c,h,w)
        features_raw = features_raw.reshape(b,t,C,D,H,W)
        
        if not self.config.train.use_gt_pose:
            # predict relative poses
            poses_cam, conf = self.encoder_traj(features_raw)
            tmp = torch.zeros_like(poses_cam)
            tmp[:,:4] = F.normalize(poses_cam[:,:4])
            tmp[:,4:] = poses_cam[:,4:]
            poses_cam = tmp
            camPoseRel_cv2 = self.encoder_traj.toSE3(poses_cam)              # [b*(t-1),4,4], relative cam pose in cv2 frame
            
            # get camera extrinsics and pose
            canonical_pose_cv2 = dataset.get_canonical_pose_cv2(device=device)                # [4,4]
            canonical_extrinsics_cv2 = dataset.get_canonical_extrinsics_cv2(device=device)
            camPoses_cv2 = canonical_pose_cv2.unsqueeze(0) @ camPoseRel_cv2
            camE_cv2 = torch.inverse(camPoses_cv2)                                            # [b*(t-1),4,4], canonicalized extrinsics
            camE_cv2 = camE_cv2.reshape(b,t-1,4,4)
            camPoses_cv2 = camPoses_cv2.reshape(b,t-1,4,4)
            camPoses_cv2 = torch.cat([canonical_pose_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camPoses_cv2], dim=1)
            camE_cv2 = torch.cat([canonical_extrinsics_cv2.reshape(1,1,4,4).repeat(b,1,1,1), camE_cv2], dim=1)  # [b,t,4,4]

            poses_cam_gt = sample['cam_poses_rel_cv2'][:,1:].to(device).reshape(b*(t-1),4,4)
            poses_cam_gt = geo_utils.mat2quat(poses_cam_gt)
            camPose_return = {'gt': poses_cam_gt, 'pred': poses_cam, 'conf': conf}
        else:
            camPoseRel_cv2 = None
            # use ground truth poses
            if self.config.train.canonicalize:
                camE_cv2 = sample['cam_extrinsics_cv2_canonicalized'].to(device)      # [b,t,4,4]
                camPoses_cv2 = sample['cam_poses_cv2_canonicalized'].to(device)       # [b,t,4,4], relative to first frame
            else:
                camE_cv2 = sample['cam_extrinsics_cv2'].to(device)                    # [b,t,4,4]
                camPoses_cv2 = sample['cam_poses_cv2'].to(device)
            camPose_return = None
        
        # parse camera for rendering
        camE_cv2 = camE_cv2.repeat(1,2,1,1)                         # [b,2*t,4,4]
        camPoses_cv2 = camPoses_cv2.repeat(1,2,1,1)                 # [b,2*t,4,4]
        camK = sample['K_cv2'].repeat(1,2,1,1)                      # [b,2*t,3,3]
        
        cameras = {
            'R': camE_cv2.reshape(b*2*t,4,4)[:,:3,:3],                # [b*2t,3,3]
            'T': camE_cv2.reshape(b*2*t,4,4)[:,:3,3],                 # [b*2t,3]
            'K': camK.reshape(b*2*t,3,3).to(device)                   # [b*2t,3,3]
        }

        if self.config.train.parameter == 'pose':
            # only return pose for larger batch size
            cameras['K'] /= 2.0
            cameras['K'][:,-1,-1] = 1.0
            cameras = cameras_from_opencv_projection(R=cameras['R'],
                                                    tvec=cameras['T'], 
                                                    camera_matrix=cameras['K'],
                                                    image_size=torch.tensor([self.config.dataset.img_size//2]*2).unsqueeze(0).repeat(b*2*t,1)).to(device)
            origin = torch.zeros(1,3).to(device)
            origin_proj = cameras.transform_points_screen(origin, eps=1e-6).squeeze()
            return camPose_return, 2 * origin_proj / self.config.dataset.img_size


        # transform features using camera poses
        features_transformed = self.rotate(voxels=features_raw,
                                           camPoses_cv2=camPoses_cv2[:,:t], 
                                           grid_size=D)                         # [b,t,C=128,D=32,H,W]
        
        # get prediction using features of some views
        # 1) features of first 3 views and render using last two cams
        # 2) features of last 2 views and render using first 3 cams
        features_3v = self.encoder_3d.fuse(features_transformed[:,:3])          # [b,C=128,D=32,H,W]
        features_2v = self.encoder_3d.fuse(features_transformed[:,-2:])
        densities_3v2v = self.encoder_3d.get_density3D(torch.cat([features_3v, features_2v], dim=0))    # [2*b,C=1,D=64,H,W]
        features_3v2v = self.encoder_3d.get_render_features(torch.cat([features_3v, features_2v], dim=0))  # [2*b,C=16,D=64,H,W]
        features_3v = features_3v2v[:b].unsqueeze(1).repeat(1,2,1,1,1,1)        
        features_2v = features_3v2v[b:].unsqueeze(1).repeat(1,3,1,1,1,1)
        features_3v2v = torch.cat([features_2v, features_3v], dim=1)
        densities_3v = densities_3v2v[:b].unsqueeze(1).repeat(1,2,1,1,1,1)
        densities_2v = densities_3v2v[b:].unsqueeze(1).repeat(1,3,1,1,1,1)
        densities_3v2v = torch.cat([densities_2v, densities_3v], dim=1)
        
        # get prediction using multi-view information
        features_mv = self.encoder_3d.fuse(features_transformed)                # [b,t,C=128,D=32,H,W] -> [b,C,D,H,W]
        densities_mv = self.encoder_3d.get_density3D(features_mv)               # [b,1,D=64,H,W]
        features_mv = self.encoder_3d.get_render_features(features_mv)          # [b,C=16,D=64,H,W]

        # render
        _, C2, D2, H2, W2 = features_mv.shape
        features_all = features_mv.unsqueeze(1).repeat(1,t,1,1,1,1)
        features_all = torch.cat([features_3v2v, features_all], dim=1).reshape(b*2*t,C2,D2,H2,W2)    # [b,2*t,C,D,H,W] -> [b*2*t,C,D,H,W]
        densities_all = densities_mv.unsqueeze(1).repeat(1,t,1,1,1,1)
        densities_all = torch.cat([densities_3v2v, densities_all], dim=1).reshape(b*2*t,1,D2,H2,W2)
        if self.config.dataset.name == 'omniobject3d':
            densities_all = densities_all.clamp(min=0.0, max=1.0)

        rendered_imgs, rendered_masks, origin_proj = self.render(cameras, features_all, densities_all, return_origin_proj=True)

        if self.config.train.use_gt_pose:
            return rendered_imgs, rendered_masks
        else:
            return rendered_imgs, rendered_masks, 2 * origin_proj / self.config.dataset.img_size, camPose_return