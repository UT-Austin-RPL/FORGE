import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import itertools
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import argparse

from config.config import config, update_config
from utils import exp_utils, train_utils
from models.model import FORGE as ReconModel
from models.perceptual_loss import VGGPerceptualLoss as PerceptualLoss
from dataset.kubric import Kubric
from dataset.omniobject3d import Omniobject3D
from scripts.kubric_trainer import train_epoch
from scripts.kubric_compute_loss import compute_pose_loss, compute_all_loss_nvs
from scripts.kubric_validation import validate
 

def set_model_train(model, config):
    '''
    The model (with both 3D-based and 2D-based pose estimator) has three training steps
        1. mode 'pose_head': train pose head fusing 2D and 3D pose estimator features
        2. mode 'pose': train the two pose estimator which uses 3D features as inputs
            only includes pose estimator parameters
        3. mode 'joint': jointly tune the model (encoder backbone and pose estimator 2D is fixed)
    '''
    model.eval()
    if config.train.parameter == 'pose_head':
        model.module.pose_head.train()
    elif config.train.parameter == 'pose':
        model.module.pose_head.train()
        model.module.encoder_traj.train()
        model.module.encoder_traj_2d.train()
    elif config.train.parameter == 'joint':
        model.module.pose_head.train()
        model.module.encoder_traj.train()
        model.module.encoder_3d.fusion_feature.train()
        model.module.encoder_3d.density_head.train()
        model.module.render.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train FORGE with only 3D pose estimator')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--local_rank', default=-1, type=int, help='node rank for distributed training')
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    # Get args and config
    args = parse_args()
    logger, output_dir, _ = exp_utils.create_logger(config, args.cfg, phase='train')
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
    if device == torch.device("cuda"):
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)

    # get model
    model = ReconModel(config).to(device)
    perceptual_loss = PerceptualLoss().to(device)
    
    # get loss function and parameters, load model weights
    assert config.train.parameter in ['pose_head', 'pose', 'joint']
    if config.train.parameter == 'pose_head':
        loss_func = compute_pose_loss
        param = model.pose_head.parameters()
        model = exp_utils.load_pose2d(model, 
                                      resume_root='./output/omniobject3d/pred_pose_2d/pred_pose_2d_2', 
                                      cpt_name='cpt_last_rot_10.010791362009114_trans_0.468529082261599.pth.tar')
        model = exp_utils.load_pose3d(model, 
                                      resume_root='./output/omniobject3d/pred_pose_3d/pred_pose_3d',
                                      cpt_name='cpt_best_rot_10.7675265518858.pth.tar')
        model = exp_utils.load_encoder_pretrained(model, 
                                                  resume_root='./output/omniobject3d/gt_pose/gt_pose',
                                                  cpt_name='cpt_last.pth.tar', strict=True)
    if config.train.parameter == 'pose':
        loss_func = compute_pose_loss
        param = list(model.pose_head.parameters()) + \
                list(model.encoder_traj_2d.parameters()) + \
                list(model.encoder_traj.parameters())
        # model = exp_utils.load_model_full(model, 
        #                                   resume_root='output/kubric/general_seq_camera_predPos_2d3d/pred_pose_quat_2d_3d_2_dropout0.5_avg_sequential_regu',
        #                                   cpt_name='cpt_best_rot.pth.tar')
        model = exp_utils.load_model_full(model, 
                                          resume_root='./output/omniobject3d/pretrain_pose_2d3d/pred_pose_2d3d_pretrain_p0.5_2',
                                          cpt_name='cpt_best_rot_10.12705252127444_psnr_0.0.pth.tar')
    elif config.train.parameter == 'joint':
        # jointly train the whole model (except backbone)
        loss_func = compute_all_loss_nvs
        param = list(model.encoder_traj.parameters()) + \
                list(model.pose_head.parameters()) + \
                list(model.encoder_3d.fusion_feature.parameters()) + \
                list(model.encoder_3d.density_head.parameters()) + \
                list(model.render.parameters())
        model = exp_utils.load_model_without_fusion(model, 
                                                    resume_root='./output/omniobject3d/pred_pose_2d3d/pred_pose_2d3d_2',
                                                    cpt_name='cpt_best_rot_8.807963791203182_psnr_0.0.pth.tar')
        model = exp_utils.load_encoder_pretrained(model, 
                                                  resume_root='./output/omniobject3d/gt_pose/gt_pose',
                                                  cpt_name='cpt_last.pth.tar', strict=True)
    
    # get optimizer
    optimizer = torch.optim.Adam(param,
                                 lr=config.train.lr * config.train.accumulation_step,
                                 weight_decay=config.train.weight_decay)
    # resume training
    best_psnr, best_rot, ep_resume = 0, float('inf'), None
    if config.train.resume:
        model, optimizer, ep_resume, best_psnr, best_rot = exp_utils.resume_training(model, optimizer, output_dir,
                                                                cpt_name='cpt_last.pth.tar')

    # distributed training
    model =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        perceptual_loss = torch.nn.parallel.DistributedDataParallel(perceptual_loss, device_ids=[args.local_rank], find_unused_parameters=True)
        device_num = len(device_ids)

    # get dataset and dataloader    
    train_data = Omniobject3D(config, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.train.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=True,
                                               sampler=train_sampler)
    val_data = Omniobject3D(config, split='val')
    val_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=config.test.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=False)    
 
    start_ep = ep_resume if ep_resume is not None else 0
    end_ep = int(config.train.total_iteration / len(train_loader)) + 1

    # train
    for epoch in range(start_ep, end_ep):
        train_sampler.set_epoch(epoch)
        train_epoch(config, 
                    loader=train_loader,
                    dataset=train_data, 
                    model=model, 
                    optimizer=optimizer,
                    epoch=epoch, 
                    output_dir=output_dir, 
                    device=device,
                    rank=args.local_rank,
                    perceptual_loss=perceptual_loss,
                    loss_func=loss_func,
                    set_model_train=set_model_train,
                    recon_sv_mv=False)

        if args.local_rank == 0:
            train_utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, 
                    checkpoint=output_dir, filename="cpt_last.pth.tar")

        # validation
        if epoch % (config.train.batch_size) == 0:
            print('Doing validation...')
            cur_psnr, cur_rot, return_dict = validate(config, 
                    loader=val_loader,
                    dataset=val_data,
                    model=model,
                    epoch=epoch, 
                    output_dir=output_dir, 
                    device=device,
                    rank=args.local_rank)
            torch.cuda.empty_cache()
            
            if config.train.parameter == 'joint' and (cur_psnr > best_psnr):
                best_psnr = cur_psnr
                if args.local_rank == 0:
                    train_utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_psnr': best_psnr,
                        'eval_dict': return_dict,
                    }, 
                    checkpoint=output_dir, filename="cpt_best_psnr_{}_rot_{}.pth.tar".format(best_psnr, cur_rot))

            if cur_rot < best_rot:
                best_rot = cur_rot
                if args.local_rank == 0:
                    train_utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_rot': best_rot,
                        'eval_dict': return_dict,
                    }, 
                    checkpoint=output_dir, filename="cpt_best_rot_{}_psnr_{}.pth.tar".format(best_rot, cur_psnr))
            
            if args.local_rank == 0:
                logger.info('Best PSNR: {} (current {}), best rot: {} (current {})'.format(best_psnr, cur_psnr, best_rot, cur_rot))


if __name__ == '__main__':
    main()