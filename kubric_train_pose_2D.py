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

from scripts.kubric_trainer_pose2D import train_epoch, validate
from models.pose_estimator_2d import PoseEstimator2D
from dataset.kubric import Kubric
 

def parse_args():
    parser = argparse.ArgumentParser(description='Train pose estimator using 2D inputs')
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
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
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
    model = PoseEstimator2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.train.lr * config.train.accumulation_step,
                                 weight_decay=config.train.weight_decay)
    best_rot, ep_resume = float('inf'), None
    if config.train.resume:
        model, optimizer, ep_resume, best_psnr, best_rot = exp_utils.resume_training(model, optimizer, output_dir,
                                                                    cpt_name='cpt_best_rot_11.268242299860368.pth.tar')

    # distributed training
    model =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        device_num = len(device_ids)

    # get dataset and dataloader    
    train_data = Kubric(config, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.train.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=True,
                                               sampler=train_sampler)
    val_data = Kubric(config, split='test')
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
                    rank=args.local_rank)

        if args.local_rank == 0:
            train_utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, 
                    checkpoint=output_dir, filename="cpt_last.pth.tar")

        if epoch % 2 == 0:
            print('Testing..')
            cur_rot, cur_trans, return_dict = validate(config, 
                    loader=val_loader,
                    dataset=val_data,
                    model=model,
                    epoch=epoch, 
                    output_dir=output_dir, 
                    device=device,
                    rank=args.local_rank)
            torch.cuda.empty_cache()
            
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
                    checkpoint=output_dir, filename="cpt_best_rot_{}_trans_{}.pth.tar".format(best_rot, cur_trans))
            
            if args.local_rank == 0:
                logger.info('Best rot error: {} (current {})'.format(best_rot, cur_rot))


if __name__ == '__main__':
    main()