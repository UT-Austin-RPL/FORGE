import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from utils import train_utils
from pytorch3d.structures import Volumes


class Rotate_world(nn.Module):
    '''
    Rotate the voxel in world coordinate using camera relative poses
    TODO: Fix the handcrafted definition of grid
    '''
    def __init__(self, config):
        super(Rotate_world, self).__init__()
        self.padding_mode = config.network.padding_mode

        self.grid_size = 32
        self.vol_size = config.render.volume_size   # 1.0
        self.single_voxel_size = self.vol_size / self.grid_size

        self.grid_coord = self._compute_grid(self.grid_size, self.single_voxel_size)  # world location of each voxel, in shape [D, H, W, 3]
        self.grid_coord_max = self.grid_coord.max().item()  # volume half size, should be 0.4844

        self.grid_coord_16 = self._compute_grid(16, self.vol_size / 16)
        self.grid_coord_max_16 = self.grid_coord_16.max().item()

        self.grid_coord_64 = self._compute_grid(64, self.vol_size / 64)
        self.grid_coord_max_64 = self.grid_coord_64.max().item()

        self.grid_coord_128 = self._compute_grid(128, self.vol_size / 128)
        self.grid_coord_max_128 = self.grid_coord_128.max().item()

        self.grid_coord_48 = self._compute_grid(48, self.vol_size / 48)
        self.grid_coord_max_48 = self.grid_coord_48.max().item()

        self.conv3d_1 = nn.Conv3d(16,16,3,padding=1)
        self.conv3d_2 = nn.Conv3d(16,16,3,padding=1)
        train_utils.normal_init(self.conv3d_1, mean=0.0, std=0.01, bias=0)
        train_utils.normal_init(self.conv3d_2, mean=0.0, std=0.01, bias=0)

        self.conv3d_3 = nn.Conv3d(128,128,3,padding=1)
        self.conv3d_4 = nn.Conv3d(128,128,3,padding=1)
        train_utils.normal_init(self.conv3d_3, mean=0.0, std=0.01, bias=0)
        train_utils.normal_init(self.conv3d_4, mean=0.0, std=0.01, bias=0)


    def _compute_grid(self, grid_size, single_voxel_size):
        density = torch.zeros(1, 1, grid_size, grid_size, grid_size)
        volume = Volumes(densities=density, voxel_size=single_voxel_size)
        grid_coord = volume.get_coord_grid(world_coordinates=True)
        return grid_coord.squeeze()    # [D, H, W, 3]


    def compute_sample_grid(self, voxels_position_camera, grid_coord_max):
        '''
        voxels_position_camera: in shape [B,N=D*H*W,3], in each camera's frame, in xyz sequence
        return: sample_grid: in shape [B,N,3], values should be in [-1,1], otherwise will be padded (zero padding)
        '''
        sample_grid = voxels_position_camera / grid_coord_max
        return sample_grid
    
    
    def get_transformation(self, camPoses_cv2):
        '''
        camPoses_cv2: in shape [B,t,4,4]
        Suppose we have two camera, as camera_1 and camera_0, we know their camera poses as T1 and T0
        For a point in world frame, denoted as P, we want apply a transformation T on it to get a P',
        so that the relative position of P in cam_0,
                the relative positino of P' in cam_1 are the same
                (rotating objects so they looks same in different cameras)
        Thus, we have: T0.inv() @ P = T1.inv() @ P' = T1.inv() @ T @ P
        We get: T = T1 @ T0.inv()
        Now we want to transform the points in world to the canonical volume of cam_2, we should use T.inv()
        P.S., originally we should use cam poses in torch3d frame, but T_torch3d = T_cv2 @ T_cv2Totorch3d,
        so it cancelled out in T1 @ T0.inv()
        '''
        B, t, _, _ = camPoses_cv2.shape

        pose_0 = camPoses_cv2[:,0:1]            # [B,1,4,4]
        pose_0 = pose_0.repeat(1,t-1,1,1)       # [B,t-1,4,4]
        pose_0 = pose_0.reshape(B*(t-1),4,4)    # [B*(t-1),4,4]

        pose_1 = camPoses_cv2[:,1:]             # [B,t-1,4,4]
        pose_1 = pose_1.reshape(B*(t-1),4,4)     # [B*(t-1),4,4]

        #T = pose_1 @ torch.inverse(pose_0)      # [B*(t-1),4,4]
        T = pose_0 @ torch.inverse(pose_1)
        return T

    
    def forward(self, voxels, camPoses_cv2, grid_size=32):
        '''
        Transform features to camera_1's frame using canonicalized camera poses
        World frame here is the camera_1's frame
        voxels: volume features of all frames, in shape [B,t,C,D,H,W]
                to be transformed into world frame
        camPoses_cv2: camera poses in opencv frame, in shape [B,t,4,4]
                P^W = T @ P^C, T is the camera pose (transform point from camera to world)
        Function:
                1) transform voxels into the coordinate of following frames
                2) calculate the sample grid in [-1,1]
                3) sample the values of voxels at corresponding sample grid location
                4) get the voxels of following frames tranformed into the world volume
        '''
        B, t, C, D, H, W = voxels.shape
        device = voxels.device

        if grid_size == 32:
            grid_coord = self.grid_coord
            grid_coord_max = self.grid_coord_max
        elif grid_size == 16:
            grid_coord = self.grid_coord_16
            grid_coord_max = self.grid_coord_max_16
        elif grid_size == 64:
            grid_coord = self.grid_coord_64
            grid_coord_max = self.grid_coord_max_64
        elif grid_size == 128:
            grid_coord = self.grid_coord_128
            grid_coord_max = self.grid_coord_max_128
        elif grid_size == 48:
            grid_coord = self.grid_coord_48
            grid_coord_max = self.grid_coord_max_48

        T = self.get_transformation(camPoses_cv2)                                               # [B*(t-1),4,4]

        voxels_position_world = grid_coord.unsqueeze(0).repeat(B*(t-1), 1, 1, 1, 1)             # [B*(t-1),D,H,W,3]
        ones = torch.ones(B*(t-1),D,H,W,1).to(voxels_position_world)                            # [B*(t-1),D,H,W,1]
        voxels_position_world = torch.cat([voxels_position_world, ones], dim=-1).to(device)     # [B*(t-1),D,H,W,4]
        voxels_position_world = voxels_position_world.reshape(B*(t-1),-1,4)                     # [B*(t-1),N,4]

        voxels_position_camera = torch.matmul(voxels_position_world, T.permute(0,2,1))          # [B*(t-1),N,4]
        voxels_position_camera = voxels_position_camera[:,:,:3]                                 # [B*(t-1),N,3]

        sample_grid = self.compute_sample_grid(voxels_position_camera, grid_coord_max).reshape(B*(t-1),D,H,W,3) # [B*(t-1),D,H,W,3]
        
        voxels_transformed = F.grid_sample(voxels[:,1:].reshape(B*(t-1),C,D,H,W), sample_grid, 
                                           padding_mode='zeros')#self.padding_mode)                      # [B*(t-1),C,D,H,W]
        
        voxels_transformed = voxels_transformed.reshape(B,t-1,C,D,H,W)
        voxels_transformed = torch.cat([voxels[:,0:1], voxels_transformed], dim=1)              # [B,t,C,D,H,W] 

        if grid_size == 32:
            # only refine rendering features
            voxels_transformed = voxels_transformed.reshape(B*t,C,D,H,W)
            # voxels_transformed = F.leaky_relu(self.conv3d_1(voxels_transformed))
            # voxels_transformed = F.leaky_relu(self.conv3d_2(voxels_transformed))
            voxels_transformed = voxels_transformed.reshape(B,t,C,D,H,W)
        elif grid_size == 16:
            # only refine rendering features
            voxels_transformed = voxels_transformed.reshape(B*t,C,D,H,W)
            # voxels_transformed = F.leaky_relu(self.conv3d_3(voxels_transformed))
            # voxels_transformed = F.leaky_relu(self.conv3d_4(voxels_transformed))
            voxels_transformed = voxels_transformed.reshape(B,t,C,D,H,W)

        return voxels_transformed


    def get_transformation_abs_pose(self, camPoses_cv2, canonicalPose_cv2):
        '''
        camPoses_cv2: in shape [B,t,4,4]
        The function is same with get_transformation, but use GT absolute pose to aggregate information in object's canonical frame
        Suppose we have a camera in the world, with pose T = [R|t], we have P^W = T @ P^C
        So the transformation is T.inv(), but also need to consider canonical pose
        '''
        b, t, _, _ = camPoses_cv2.shape
        device = camPoses_cv2.device

        pose_0 = canonicalPose_cv2.unsqueeze(0).repeat(b*t,1,1).to(device)

        T = pose_0 @ torch.inverse(camPoses_cv2.reshape(b*t,4,4))
        return T
    
    
    def transform_with_abs_pose(self, voxels, camPoses_cv2, canonicalPose_cv2):
        '''
        Transform features to object's canonical frame using absolute camera poses
        World frame here is object's canonical space
        voxels: volume features of all frames, in shape [B,t,C,D,H,W]
                to be transformed into world frame
        camPoses_cv2: camera poses in opencv frame, in shape [B,t,4,4]
                P^W = T @ P^C, T is the camera pose (transform point from camera to world)
        canonicalPose_cv2: pre-defined canonical pose
        Function:
                1) transform voxels into the coordinate of all frames
                2) calculate the sample grid in [-1,1]
                3) sample the values of voxels at corresponding sample grid location
                4) get the voxels of all frames tranformed into the world volume
        '''
        B, t, C, D, H, W = voxels.shape
        device = voxels.device

        T = self.get_transformation_abs_pose(camPoses_cv2, canonicalPose_cv2)                   # [B*t,4,4]

        voxels_position_world = self.grid_coord.unsqueeze(0).repeat(B*t, 1, 1, 1, 1)            # [B*t,D,H,W,3]
        ones = torch.ones(B*t,D,H,W,1).to(voxels_position_world)                                # [B*t,D,H,W,1]
        voxels_position_world = torch.cat([voxels_position_world, ones], dim=-1).to(device)     # [B*t,D,H,W,4]
        voxels_position_world = voxels_position_world.reshape(B*t,-1,4)                         # [B*t,N,4]

        voxels_position_camera = torch.matmul(voxels_position_world, T.permute(0,2,1))          # [B*t,N,4]
        voxels_position_camera = voxels_position_camera[:,:,:3]                                 # [B*t,N,3]

        sample_grid = self.compute_sample_grid(voxels_position_camera).reshape(B*t,D,H,W,3)     # [B*t,D,H,W,3]
        
        voxels_transformed = F.grid_sample(voxels.reshape(B*t,C,D,H,W), sample_grid, 
                                           padding_mode='zeros')                                # [B*t,C,D,H,W]

        voxels_transformed = F.leaky_relu(self.conv3d_1(voxels_transformed))
        voxels_transformed = F.leaky_relu(self.conv3d_2(voxels_transformed))
        voxels_transformed = voxels_transformed.reshape(B,t,C,D,H,W)            

        return voxels_transformed


    def get_transformation_every2_pose(self, camPoseEvery2_cv2, cv2_to_torch3d):
        '''
        camPoseEvery2_cv2 is T^c1_c1Toc2, is the pose of camera 2 in camera 1's frame, in shape
        we have P^c1 = T^c1_c1Toc2 @ P^c2
        We want to transform point in camera 1 to camera 2's frame
        Thus, the transformation should be T^c2_c2Toc1 = T^c1_c1Toc2.inv()
        '''
        b, t, _, _ = camPoseEvery2_cv2.shape
        device = camPoseEvery2_cv2.device

        cv2_to_torch3d = cv2_to_torch3d.unsqueeze(0).to(device)
        camPoseEvery2_torch3d = camPoseEvery2_cv2.reshape(b*t,4,4) @ cv2_to_torch3d
        return torch.inverse(camPoseEvery2_torch3d)

    
    def transform_with_every2_pose(self, voxels, camPoseEvery2_cv2, cv2_to_torch3d):
        '''
        Suppose we have N views, we transform the features of first N-1 cameras to last N-1 camera's space
        World frame here is each of the coordinate space of first N-1 cameras
        voxels: volume features of first N-1 frames, in shape [B,t,C,D,H,W]
                to be transformed into last N-1 frames
        camPoseEvery2_cv2: relative camera poses in opencv frame, in shape [B,t,4,4], are T^c1_c1Toc2
                P^c1 = T^c1_c1Toc2 @ P^c2
        '''
        B, t, C, D, H, W = voxels.shape
        device = voxels.device

        T = self.get_transformation_every2_pose(camPoseEvery2_cv2, cv2_to_torch3d)              # [B*t,4,4]

        voxels_position_cam1 = self.grid_coord.unsqueeze(0).repeat(B*t, 1, 1, 1, 1)             # [B*t,D,H,W,3]
        ones = torch.ones(B*t,D,H,W,1).to(voxels_position_cam1)                                 # [B*t,D,H,W,1]
        voxels_position_cam1 = torch.cat([voxels_position_cam1, ones], dim=-1).to(device)       # [B*t,D,H,W,4]
        voxels_position_cam1 = voxels_position_cam1.reshape(B*t,-1,4)                           # [B*t,N,4]

        voxels_position_cam2 = torch.matmul(voxels_position_cam1, T.permute(0,2,1))             # [B*t,N,4]
        voxels_position_cam2 = voxels_position_cam2[:,:,:3]                                     # [B*t,N,3]

        sample_grid = self.compute_sample_grid(voxels_position_cam2).reshape(B*t,D,H,W,3)       # [B*t,D,H,W,3]
        
        voxels_transformed = F.grid_sample(voxels.reshape(B*t,C,D,H,W), sample_grid, 
                                           padding_mode='zeros')                                # [B*t,C,D,H,W]

        voxels_transformed = F.leaky_relu(self.conv3d_1(voxels_transformed))
        voxels_transformed = F.leaky_relu(self.conv3d_2(voxels_transformed))
        voxels_transformed = voxels_transformed.reshape(B,t,C,D,H,W)            

        return voxels_transformed

        
