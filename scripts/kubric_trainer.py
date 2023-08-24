import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
import os
import dataclasses
import torch.distributed as dist
import itertools
from utils import train_utils, vis_utils, geo_utils, exp_utils, eval_utils

logger = logging.getLogger(__name__)


def train_epoch(config, loader, dataset, model, optimizer, 
                epoch, output_dir, device, rank, 
                perceptual_loss=None, loss_func=None, set_model_train=None, recon_sv_mv=True):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()

    if config.dataset.name == 'kubric':
        max_norm = 10.0
    elif config.dataset.name == 'omniobject3d':
        max_norm = 5.0

    set_model_train(model, config)
    if perceptual_loss is not None:
        perceptual_loss.eval()

    adjust_iter_num = config.train.adjust_iter_num

    batch_end = time.time()
    for batch_idx, sample in enumerate(loader):

        # adjust lr
        iter_num = batch_idx + len(loader) * epoch
        if iter_num in adjust_iter_num:
            train_utils.adjust_lr(config, optimizer, iter_num, adjust_iter_num)
        
        #sample = train_utils.dict_to_cuda(sample)
        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        losses = {}
        # reconstruction loss
        loss_recon, losses, rendered_imgs, rendered_masks = loss_func(config, epoch, sample, dataset, model, losses, device, perceptual_loss)
        time_meters.add_loss_value('Reconstruction time', time.time() - end)
        end = time.time()

        total_loss = loss_recon

        dist.barrier()

        total_loss = total_loss / config.train.accumulation_step
        total_loss.backward()
        if (batch_idx+1) % config.train.accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad()

        for k, v in losses.items():
            if v is not None:
                loss_meters.add_loss_value(k, v)
        
        time_meters.add_loss_value('Batch time', time.time() - batch_end)

        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, recon {recon_time:.3f}s, all {batch_time:.3f}s, Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].val,
                recon_time=time_meters.average_meters['Reconstruction time'].val,
                batch_time=time_meters.average_meters['Batch time'].val
            )
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.6f} ({loss.avg:.6f}), '.format(
                        k, loss=loss_meters.average_meters[k]
                )
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)

        
        if iter_num % config.vis_freq == 0 and rank == 0 and torch.is_tensor(rendered_imgs):
                if recon_sv_mv:
                    vis_utils.vis_seq_sv_mv(vid_clips=sample['images'],
                                    vid_masks=sample['fg_probabilities'],
                                    recon_clips=rendered_imgs,
                                    recon_masks=rendered_masks,
                                    iter_num=iter_num,
                                    output_dir=output_dir,
                                    subfolder='train_seq')
                else:
                    vis_utils.vis_seq(vid_clips=sample['images'],
                                    vid_masks=sample['fg_probabilities'],
                                    recon_clips=rendered_imgs,
                                    recon_masks=rendered_masks,
                                    iter_num=iter_num,
                                    output_dir=output_dir,
                                    subfolder='train_seq')

        batch_end = time.time()