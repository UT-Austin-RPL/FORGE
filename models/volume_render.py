import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Volumes
from pytorch3d.renderer import VolumeRenderer, NDCGridRaysampler, EmissionAbsorptionRaymarcher
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.renderer.cameras import PerspectiveCameras


class VolRender(nn.Module):
    def __init__(self, config):
        super(VolRender, self).__init__()
        self.img_size = config.dataset.img_size
        self.volume_physical_size = config.render.volume_size

        # first render a featue map with half resolution, then upsample
        self.raySampler = NDCGridRaysampler(image_width=self.img_size//2,
                                            image_height=self.img_size//2,
                                            n_pts_per_ray=config.render.n_pts_per_ray,
                                            min_depth=config.render.min_depth,
                                            max_depth=config.render.max_depth)
        self.rayMarcher = EmissionAbsorptionRaymarcher()
        self.renderer = VolumeRenderer(raysampler=self.raySampler, raymarcher=self.rayMarcher)

        # from rendered feature map to rgb
        self.k_size = config.render.k_size
        self.pad_size = self.k_size // 2
        self.conv_rgb = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=self.k_size+1, stride=2, padding=self.pad_size),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=self.k_size, stride=1, padding=self.pad_size),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 3, kernel_size=self.k_size, stride=1, padding=self.pad_size),
        )
        

    def forward(self, camera_params, feature_3d, density_3d, render_depth=False, return_origin_proj=False):
        '''
        cameras: pytorch3d perspective camera, parameters in batch size B
        feature_3d: [B,C,D,H,W]
        density_3d: [B,1,D,H,W]
        camera_pose: should be in [B,3,4]
        '''
        B, C, D, H, W = feature_3d.shape
        device = feature_3d.device

        camera_params['K'] /= 2.0
        camera_params['K'][:,-1,-1] = 1.0

        cameras = cameras_from_opencv_projection(R=camera_params['R'],
                                                 tvec=camera_params['T'], 
                                                 camera_matrix=camera_params['K'],
                                                 image_size=torch.tensor([self.img_size//2]*2).unsqueeze(0).repeat(B,1)).to(device)
        
        single_voxel_size = self.volume_physical_size / D
        volume = Volumes(densities=density_3d,
                         features=feature_3d,
                         voxel_size=single_voxel_size)

        rendered = self.renderer(cameras=cameras, volumes=volume, render_depth=render_depth)[0]  # [B,H,W,C=32+1]
        if not render_depth:
            rendered_imgs, rendered_silhouettes = rendered.split([C, 1], dim=-1)
        else:
            rendered_imgs, rendered_silhouettes, rendered_depth = rendered.split([C, 1, 1], dim=-1)
            rendered_depth = rendered_depth.permute(0,3,1,2).contiguous()
            rendered_depth = F.upsample(rendered_depth, size=[self.img_size]*2, mode='bilinear')

        rendered_imgs = rendered_imgs.permute(0,3,1,2).contiguous()
        rendered_silhouettes = rendered_silhouettes.permute(0,3,1,2).contiguous()
        rendered_imgs = F.relu(self.conv_rgb(rendered_imgs))
        rendered_silhouettes = F.upsample(rendered_silhouettes, size=[self.img_size]*2, mode='bilinear')

        if return_origin_proj:
            origin = torch.zeros(1,3).to(device)
            origin_proj = cameras.transform_points_screen(origin, eps=1e-6).squeeze()
            if render_depth:
                return rendered_imgs, rendered_silhouettes, rendered_depth, origin_proj[:,:2]
            else:
                return rendered_imgs, rendered_silhouettes, origin_proj[:,:2]
        else:
            if render_depth:
                return rendered_imgs, rendered_silhouettes, rendered_depth
            else:
                return rendered_imgs, rendered_silhouettes

    
    def proj_origin(self, camera_params, device):
        B = camera_params['K'].shape[0]

        camera_params['K'] /= 2.0
        camera_params['K'][:,-1,-1] = 1.0

        cameras = cameras_from_opencv_projection(R=camera_params['R'],
                                                 tvec=camera_params['T'], 
                                                 camera_matrix=camera_params['K'],
                                                 image_size=torch.tensor([self.img_size//2]*2).unsqueeze(0).repeat(B,1)).to(device)
        origin = torch.zeros(1,3).to(device)
        origin_proj = cameras.transform_points_screen(origin, eps=1e-6).squeeze()
        return origin_proj[:,:2]    # in pixel space
         


