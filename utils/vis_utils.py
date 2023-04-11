import numpy as np
import torch
import cv2
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.patches as Patch
import imageio
from einops import rearrange
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def vis_single_frame(imgs, rendered_imgs,
                     masks, rendered_masks,
                     densities,
                     output_dir, batch_idx):
    output_dir = os.path.join(output_dir, 'visualization', 'train_encoder3d')
    os.makedirs(output_dir, exist_ok=True)
    save_name = os.path.join(output_dir, str(batch_idx) + '.jpg')

    max_item = 5
    B = imgs.shape[0]
    rows = min(max_item, B)

    imgs = imgs.detach().cpu()
    rendered_imgs = rendered_imgs.detach().cpu()
    masks = masks.detach().cpu()
    rendered_masks = rendered_masks.detach().cpu()
    densities = densities.detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 12))
    fig.clf()
    col_nb = 4  # Col 1: img, 2: rendered img, 3: mask, 4: rendered mask, 5-6: density voxel

    #pylab.subplots_adjust(left=0.15, bottom=0.1, top=0.9, right=0.95, hspace=0.4, wspace=0.8)
    for i in range(rows):
        #from IPython import embed; embed()
        img = imgs[i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 1)
        ax.imshow(img)
        ax.axis("off")

        rendered_img = rendered_imgs[i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 2)
        ax.imshow(rendered_img)
        ax.axis('off')

        mask = masks[i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 3)
        ax.imshow(mask)
        ax.axis("off")

        rendered_mask = rendered_masks[i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 4)
        ax.imshow(rendered_mask)
        ax.axis('off')

        # density = densities[i][0] > 0.3
        # #color = np.zeros(density.shape + (4,))
        # #color[..., :-1] = 1
        # #color[..., -1] = densities[i][0]  # alpha
        # ax = fig.add_subplot(rows, col_nb, i * col_nb + 5, projection='3d')
        # ax.view_init(elev=30, azim=-60)
        # ax.voxels(density, edgecolor='k')
        
        # ax = fig.add_subplot(rows, col_nb, i * col_nb + 6, projection='3d')
        # ax.view_init(elev=0, azim=-90)  # side view
        # ax.voxels(density, edgecolor='k')

    plt.savefig(save_name, dpi=200)
    plt.close('all')


def vis_single_frame_NVS(imgs, rendered_imgs,
                     masks, rendered_masks,
                     densities,
                     output_dir, batch_idx):
    output_dir = os.path.join(output_dir, 'visualization', 'train_encoder3d_NVS')
    os.makedirs(output_dir, exist_ok=True)
    save_name = os.path.join(output_dir, str(batch_idx) + '.jpg')

    max_item = 5
    B = imgs.shape[0]
    rows = min(max_item, B)

    imgs = imgs.detach().cpu()
    rendered_imgs = rendered_imgs.detach().cpu()
    masks = masks.detach().cpu()
    rendered_masks = rendered_masks.detach().cpu()
    densities = densities.detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 12))
    fig.clf()
    col_nb = 6  # Col 1: img, 2: rendered img, 3: mask, 4: rendered mask, 5-6: NVS img and mask

    #pylab.subplots_adjust(left=0.15, bottom=0.1, top=0.9, right=0.95, hspace=0.4, wspace=0.8)
    for i in range(rows):
        #from IPython import embed; embed()
        img = imgs[i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 1)
        ax.imshow(img)
        ax.axis("off")

        rendered_img = rendered_imgs[i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 2)
        ax.imshow(rendered_img)
        ax.axis('off')

        mask = masks[i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 3)
        ax.imshow(mask)
        ax.axis("off")

        rendered_mask = rendered_masks[i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 4)
        ax.imshow(rendered_mask)
        ax.axis('off')

        rendered_img_NVS = rendered_imgs[B+i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 5)
        ax.imshow(rendered_img_NVS)
        ax.axis('off')

        rendered_mask_NVS = rendered_masks[B+i].permute(1,2,0).numpy()
        ax = fig.add_subplot(rows, col_nb, i * col_nb + 6)
        ax.imshow(rendered_mask_NVS)
        ax.axis("off")

    plt.savefig(save_name, dpi=200)
    plt.close('all')


def vis_seq(vid_clips, vid_masks, recon_clips, recon_masks, iter_num, output_dir, subfolder='train_seq', vid_depths=None, recon_depths=None):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)

    vid_clips = vid_clips.detach().cpu()        # [B,t,c,h,w]
    recon_clips = recon_clips.detach().cpu()
    vid_masks = vid_masks.detach().cpu()
    recon_masks = recon_masks.detach().cpu()

    addition_col, depth_diff_col = 0, 0
    if torch.is_tensor(recon_depths):
        recon_depths = recon_depths.detach().cpu()
        addition_col += 1
    if torch.is_tensor(vid_depths):
        vid_depths = vid_depths.detach().cpu()
        addition_col += 1
    if torch.is_tensor(vid_depths) and torch.is_tensor(recon_depths):
        depth_diff_col = 1

    B = vid_clips.shape[0]
    for i in range(B):
        save_name = os.path.join(output_dir, str(iter_num) + '_' + str(i) + '.jpg')
        rows = vid_clips.shape[1]
        fig = plt.figure(figsize=(12, 12))
        fig.clf()
        col_nb = 4 + addition_col + depth_diff_col # Col 1: img, 2: rendered img, 3: mask, 4: rendered mask, 5 (5-7): depths and rendered depth

        for j in range(rows):
            img = vid_clips[i][j].permute(1,2,0).numpy()            # [h,w,c]
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 1)
            ax.imshow(img)
            ax.axis("off")

            rendered_img = recon_clips[i][j].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 2)
            ax.imshow(rendered_img)
            ax.axis('off')

            mask = vid_masks[i][j].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 3)
            ax.imshow(mask)
            ax.axis("off")

            rendered_mask = recon_masks[i][j].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 4)
            ax.imshow(rendered_mask)
            ax.axis('off')

            if torch.is_tensor(vid_depths):
                depth = vid_depths[i][j].permute(1,2,0).numpy()
                ax = fig.add_subplot(rows, col_nb, j * col_nb + col_nb - 1 - depth_diff_col)
                ax.imshow(depth, cmap='magma', vmin=0, vmax=2)
                ax.axis('off')
            
            if torch.is_tensor(recon_depths):
                rendered_depth = recon_depths[i][j].permute(1,2,0).numpy()
                ax = fig.add_subplot(rows, col_nb, j * col_nb + col_nb - depth_diff_col)
                ax.imshow(rendered_depth, cmap='magma', vmin=0, vmax=2)
                ax.axis('off')

            if torch.is_tensor(recon_depths) and torch.is_tensor(vid_depths):
                ax = fig.add_subplot(rows, col_nb, j * col_nb + col_nb)
                ax.imshow(np.abs(depth - rendered_depth), cmap='magma', vmin=0, vmax=2)
                ax.axis('off')
        
        plt.savefig(save_name, dpi=200)
        plt.close('all')


def vis_test(vid_clips, recon_clips, dir_name, output_dir, subfolder='train_seq'):
    output_dir = os.path.join(output_dir, subfolder, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    vid_clips = vid_clips.detach().cpu()        # [B,t,c,h,w]
    t_img = vid_clips.shape[1]
    recon_clips = recon_clips.detach().cpu()
    t_recon = recon_clips.shape[1]

    # save images
    for idx in range(t_img):
        cur_img = vid_clips[0,idx].permute(1,2,0).numpy() * 255.0
        save_name = os.path.join(output_dir, '{}_img.jpg'.format(idx))
        cv2.imwrite(save_name, cur_img)

    for idx in range(t_recon):
        cur_img = recon_clips[0,idx].permute(1,2,0).numpy() * 255.0
        save_name = os.path.join(output_dir, '{}_img_nvs.jpg'.format(idx + (t_img - t_recon)))
        cv2.imwrite(save_name, cur_img)


def vis_seq_sv_mv(vid_clips, vid_masks, recon_clips, recon_masks, iter_num, output_dir, inv_normalize=False, subfolder='train_seq'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)

    vid_clips = vid_clips.detach().cpu()        # [B,t,c,h,w]
    recon_clips = recon_clips.detach().cpu()
    if inv_normalize:
        b,t,c,h,w = vid_clips.shape
        b,t2,c,h,w = recon_clips.shape
        vid_clips = rearrange(vid_clips, 'b t c h w -> (b t) c h w')
        recon_clips = rearrange(recon_clips, 'b t c h w -> (b t) c h w')
        all_clips = torch.cat([vid_clips, recon_clips], dim=0)
        all_clips = inverse_normalize(all_clips)
        vid_clips, recon_clips = all_clips.split([b*t, b*t2], dim=0)
        vid_clips = rearrange(vid_clips, '(b t) c h w -> b t c h w', t=t)
        recon_clips = rearrange(recon_clips, '(b t) c h w -> b t c h w', t=t2)

    vid_masks = vid_masks.detach().cpu()        # [B,2t,c,h,w]
    recon_masks = recon_masks.detach().cpu()

    t = vid_clips.shape[1]

    B = vid_clips.shape[0]
    for i in range(B):
        save_name = os.path.join(output_dir, str(iter_num) + '_' + str(i) + '.jpg')
        rows = vid_clips.shape[1]
        fig = plt.figure(figsize=(12, 12))
        fig.clf()
        col_nb = 6  
        
        # Col 1: img, 2: rendered img sv, 3, rendered img mv, 
        #     4: mask, 5: rendered mask sv, 6, rendered mask mv

        for j in range(rows):
            img = vid_clips[i][j].permute(1,2,0).numpy()            # [h,w,c]
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 1)
            ax.imshow(img)
            ax.axis("off")

            rendered_img = recon_clips[i][j].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 2)
            ax.imshow(rendered_img)
            ax.axis('off')

            rendered_img = recon_clips[i][j+t].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 3)
            ax.imshow(rendered_img)
            ax.axis('off')

            mask = vid_masks[i][j].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 4)
            ax.imshow(mask)
            ax.axis("off")

            rendered_mask = recon_masks[i][j].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 5)
            ax.imshow(rendered_mask)
            ax.axis('off')

            rendered_mask = recon_masks[i][j+t].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 6)
            ax.imshow(rendered_mask)
            ax.axis('off')
        
        plt.savefig(save_name, dpi=200)
        plt.close('all')


def vis_NVS(imgs, masks, img_name, output_dir, inv_normalize=False, subfolder='val_seq', depths=None):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)
    save_name = os.path.join(output_dir, str(img_name) + '.gif')

    imgs = imgs.detach().cpu()  # [N,c,h,w]
    if inv_normalize:
        imgs = inverse_normalize(imgs)

    masks = masks.detach().cpu()  # [N,1,h,w]
    masks = masks.repeat(1,3,1,1)
    if torch.is_tensor(depths):
        depths = depths.detach().cpu()  # [N,1,h,w]
        depths = depths.repeat(1,3,1,1)
        imgs = 255 * torch.cat([imgs, masks, depths], dim=-1)  # [N,c,h, 3*w]
    else:
        imgs = 255 * torch.cat([imgs, masks], dim=-1)  # [N,c,h, 2*w]
        

    frames = [np.uint8(img.permute(1,2,0).numpy()) for img in imgs]  # image in [h,w,c]
    #from IPython import embed; embed()
    imageio.mimsave(save_name, frames, 'GIF', duration=0.1)


def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor



def vis_poses(vid_clips, pose, pose_gt, output_dir, save_name, vis_clips=False):
    '''
    vid_clips: [b, t, 3, h, w]
    E: [b, t, 4, 4]
    '''
    output_dir = os.path.join(output_dir, 'visualization', 'pose')
    os.makedirs(output_dir, exist_ok=True)

    vid_clips = vid_clips.detach().cpu()        # [B,t,c,h,w]
    t = vid_clips.shape[1]

    B = vid_clips.shape[0]
    for i in range(B):
        # visualize inputs
        if vis_clips:
            save_name_img = os.path.join(output_dir, str(save_name) + '_' + str(i) + '.jpg')
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(1, 1, 1)
            cur_clips = [it for it in vid_clips[i]]
            cur_clips = torch.cat(cur_clips, dim=-1)    # [c,h,w*t]
            cur_clips = cur_clips.permute(1,2,0)
            cur_clips[cur_clips.sum(dim=-1) == 0.0] = torch.tensor([1.0, 1.0, 1.0])
            ax.imshow(cur_clips.numpy())
            ax.axis('off')  
            plt.savefig(save_name_img, dpi=200)
            plt.close('all')

        # visualize pose
        save_name_pose = os.path.join(output_dir, str(save_name) + '_' + str(i) + '.jpg')
        os.makedirs('/'.join(save_name_pose.split('/')[:-1]), exist_ok=True)
        visualizer = CameraPoseVisualizer([-1.5, 1.5], [-1.5, 1.5], [-1.5,1.5], elev=0, azim=90)
        cur_pose = pose[i].detach().cpu()  # [t,4,4]
        cur_pose_gt = pose_gt[i].detach().cpu()
        #visualizer.extrinsic2pyramid(cur_pose[0].numpy(), [1,1,1], 'b', 0.3, 0.3)
        visualizer.extrinsic2pyramid(cur_pose[0].numpy(), plt.cm.rainbow(0 / t), plt.cm.rainbow(0 / t), 0.35, 0.35, 0.3)
        for it in range(1,t):
            # visualizer.extrinsic2pyramid(cur_pose[it].numpy(), [1,1,1], [0,0,0], 0.3, 0.3)
            # visualizer.extrinsic2pyramid(cur_pose_gt[it].numpy(), [1,1,1], 'r', 0.3, 0.3)
            visualizer.extrinsic2pyramid(cur_pose_gt[it].numpy(), [1,1,1], plt.cm.rainbow(it / t), 0.35, 0.35, 0.2)
            visualizer.extrinsic2pyramid(cur_pose[it].numpy(), plt.cm.rainbow(it / t), plt.cm.rainbow(it / t), 0.35, 0.35, 0.3)
        visualizer.colorbar(t)
        visualizer.save(save_name_pose)

    

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim, elev=None, azim=None):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.gca(projection='3d')
        if elev != None and azim != None:
            self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.ax.set_axis_off()

    def extrinsic2pyramid(self, extrinsic, facecolor='r', edgecolor='b', focal_len_scaled=0.35, aspect_ratio=0.35, alpha=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                            [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                            [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                            [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                            [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=facecolor, linewidths=1, edgecolors=edgecolor, alpha=alpha))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                            orientation='vertical', label='Frame Number', shrink=0.15)

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()

    def save(self, save_path):
        plt.savefig(save_path, dpi=200)
        plt.close('all')