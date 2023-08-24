import os
import numpy as np

if __name__ == '__main__':
    file_path = '/vision/hwjiang/open_resource/FORGE/output/omniobject3d/optimize/optimize/results/results.txt'

    unseen_psnr = []
    unseen_ssim = []
    unseen_lpips = []
    unseen_depth = []
    unseen_rot = []
    unseen_trans = []
    seen_psnr = []
    seen_ssim = []
    seen_lpips = []
    seen_depth = []
    seen_rot = []
    seen_trans = []

    unseen_psnr2 = []
    unseen_ssim2 = []
    unseen_lpips2 = []
    unseen_depth2 = []
    unseen_rot2 = []
    unseen_trans2 = []
    seen_psnr2 = []
    seen_ssim2 = []
    seen_lpips2 = []
    seen_depth2 = []
    seen_rot2 = []
    seen_trans2 = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        idx, seen_flag, state, psnr, ssim, lpips, rot, trans, depth = line.strip().split(', ')
        idx = int(idx.split(' ')[-1])
        # if idx in [320, 520, 600]:
        #     continue
        seen_flag = (seen_flag.split(' ')[-1] == 'True')
        psnr = float(psnr.split(' ')[-1])
        ssim = float(ssim.split(' ')[-1])
        lpips = float(lpips.split(' ')[-1])
        depth = float(depth.split(' ')[-1])
        rot = float(rot.split(' ')[-1])
        trans = float(trans.split(' ')[-1])

        if 'before' in state:
            if seen_flag == True:
                seen_psnr.append(psnr)
                seen_ssim.append(ssim)
                seen_lpips.append(lpips)
                seen_depth.append(depth)
                seen_rot.append(rot)
                seen_trans.append(trans)
            else:
                unseen_psnr.append(psnr)
                unseen_ssim.append(ssim)
                unseen_lpips.append(lpips)
                unseen_depth.append(depth)
                unseen_rot.append(rot)
                unseen_trans.append(trans)
        elif 'after' in state:
            if seen_flag == True:
                seen_psnr2.append(psnr)
                seen_ssim2.append(ssim)
                seen_lpips2.append(lpips)
                seen_depth2.append(depth)
                seen_rot2.append(rot)
                seen_trans2.append(trans)
            else:
                unseen_psnr2.append(psnr)
                unseen_ssim2.append(ssim)
                unseen_lpips.append(lpips)
                unseen_depth2.append(depth)
                unseen_rot2.append(rot)
                unseen_trans2.append(trans)

    
    unseen_psnr = np.array(unseen_psnr)
    unseen_ssim = np.array(unseen_ssim)
    unseen_lpips = np.array(unseen_lpips)
    unseen_depth = np.array(unseen_depth)
    unseen_rot = np.array(unseen_rot)
    unseen_trans = np.array(unseen_trans)
    seen_psnr = np.array(seen_psnr)
    seen_ssim = np.array(seen_ssim)
    seen_lpips = np.array(seen_lpips)
    seen_depth = np.array(seen_depth)
    seen_rot = np.array(seen_rot)
    seen_trans = np.array(seen_trans)
    print('---------------------before mean-----------------------------------')
    print('unseen: PSNR {}, ssim {}, lpips {}, depth {}'.format(unseen_psnr.mean(), unseen_ssim.mean(), unseen_lpips.mean(), unseen_depth.mean()))
    print('unseen: Rot {}, Trans {}'.format(unseen_rot.mean(), unseen_trans.mean()))
    print('seen: PSNR {}, ssim {}, lpips {}, depth {}'.format(seen_psnr.mean(), seen_ssim.mean(), seen_lpips.mean(), seen_depth.mean()))
    print('seen: Rot {}, Trans {}'.format(seen_rot.mean(), seen_trans.mean()))
    print('---------------------------------------------------------------')
    print('---------------------before median-----------------------------------')
    print('unseen: PSNR {}, ssim {}, lpips {}, depth {}'.format(np.median(unseen_psnr), np.median(unseen_ssim), np.median(unseen_lpips), np.median(unseen_depth)))
    print('unseen: Rot {}, Trans {}'.format(np.median(unseen_rot), np.median(unseen_trans)))
    print('seen: PSNR {}, ssim {}, lpips {}, depth {}'.format(np.median(seen_psnr), np.median(seen_ssim), np.median(seen_lpips), np.median(seen_depth)))
    print('seen: Rot {}, Trans {}'.format(np.median(seen_rot), np.median(seen_trans)))
    print('---------------------------------------------------------------')

    unseen_psnr2 = np.array(unseen_psnr2)
    unseen_ssim2 = np.array(unseen_ssim2)
    unseen_lpips2 = np.array(unseen_lpips2)
    unseen_depth2 = np.array(unseen_depth2)
    unseen_rot2 = np.array(unseen_rot2)
    unseen_trans2 = np.array(unseen_trans2)
    seen_psnr2 = np.array(seen_psnr2)
    seen_ssim2 = np.array(seen_ssim2)
    seen_lpips2 = np.array(seen_lpips2)
    seen_depth2 = np.array(seen_depth2)
    seen_rot2 = np.array(seen_rot2)
    seen_trans2 = np.array(seen_trans2) 
    print('---------------------after mean------------------------------------')
    print('unseen: PSNR {}, ssim {}, lpips {}, depth {}'.format(unseen_psnr2.mean(), unseen_ssim2.mean(), unseen_lpips2.mean(), unseen_depth2.mean()))
    print('unseen: Rot {}, Trans {}'.format(unseen_rot2.mean(), unseen_trans2.mean()))
    print('seen: PSNR {}, ssim {}, lpips {}, depth {}'.format(seen_psnr2.mean(), seen_ssim2.mean(), seen_lpips2.mean(), seen_depth2.mean()))
    print('seen: Rot {}, Trans {}'.format(seen_rot2.mean(), seen_trans2.mean()))
    print('---------------------------------------------------------------')
    print('---------------------after median------------------------------------')
    print('unseen: PSNR {}, ssim {}, lpips {}, depth {}'.format(np.median(unseen_psnr2), np.median(unseen_ssim2), np.median(unseen_lpips2), np.median(unseen_depth2)))
    print('unseen: Rot {}, Trans {}'.format(np.median(unseen_rot2), np.median(unseen_trans2)))
    print('seen: PSNR {}, ssim {}, lpips {}, depth {}'.format(np.median(seen_psnr2), np.median(seen_ssim2), np.median(seen_lpips2), np.median(seen_depth2)))
    print('seen: Rot {}, Trans {}'.format(np.median(seen_rot2), np.median(seen_trans2)))
    print('---------------------------------------------------------------')
    