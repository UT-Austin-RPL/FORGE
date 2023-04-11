import yaml
import os
import numpy as np
from easydict import EasyDict as edict

config = edict()

# experiment config
config.exp_name = 'co3d'
config.output_dir = './output/'
config.log_dir = './log'
config.workers = 8
config.print_freq = 100
config.vis_freq = 300
config.eval_vis_freq = 20
config.vis_density_freq = 10000
config.seed = 0

# dataset config
config.dataset = edict()
config.dataset.name = 'co3d'
config.dataset.category = 'apple'
config.dataset.task = 'multisequence'
config.dataset.img_size = 512
config.dataset.num_frame = 5
config.dataset.frame_interval = 5
config.dataset.mask_images = False
config.dataset.augmentation = False
config.dataset.train_all_frame = False
config.dataset.train_shuffle = False

# network config
config.network = edict()
config.network.backbone = 'resnet'
config.network.scale_rotate = 0.01
config.network.scale_translate = 0.01
config.network.padding_mode = 'zeros'
config.network.rot_representation = 'euler'

# render config
config.render = edict()
config.render.n_pts_per_ray = 200
config.render.volume_size = 1.0  #  in meters
config.render.min_depth = 0.1
config.render.max_depth = 1.2
config.render.camera_z = 0.6  # camera pose T_z
config.render.camera_focal = 250
config.render.k_size = 5

# loss config
config.loss = edict()
config.loss.recon_rgb = 1.0
config.loss.recon_mask = 0.2
config.loss.perceptual_img = 0.0
config.loss.regu_origin_proj = 0.0

# training config
config.train = edict()
config.train.lr = 0.0001
config.train.weight_decay = 0.0001
config.train.schedular_step = 10
config.train.schedular_gamma = 0.7
config.train.end_epoch = 100
config.train.resume = False
config.train.batch_size = 16
config.train.snapshot_freq = 10
config.train.total_iteration = 200000
config.train.sv_pretrain = ''
config.train.use_gt_pose = False
config.train.canonicalize = True
config.train.accumulation_step = 2
config.train.normalize_img = False
config.train.parameter = ''
config.train.adjust_iter_num = [0]

# test config
config.test = edict()
config.test.batch_size = 4
config.test.compute_metric = True


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                     config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

