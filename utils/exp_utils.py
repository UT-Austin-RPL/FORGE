import os
import sys
import json
import time
import datetime
import warnings
import torch
from easydict import EasyDict as edict
from pathlib import Path
import logging

def progress_bar(msg=None):
    L = []
    if msg:
        L.append(msg)

    msg = ''.join(L)
    sys.stdout.write(msg+'\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeters:
    def __init__(self):
        super().__init__()
        self.average_meters = {}

    def add_loss_value(self, loss_name, loss_val, n=1):
        if loss_name not in self.average_meters:
            self.average_meters[loss_name] = AverageMeter()
        self.average_meters[loss_name].update(loss_val, n=n)


class Monitor:
    def __init__(self, hosting_file):
        self.hosting_file = hosting_file

    def log_train(self, epoch, errors):
        log_errors(epoch, errors, self.hosting_file)


def log_errors(epoch, errors, log_path=None):
    now = time.strftime("%c")
    message = "(epoch: {epoch}, time: {t})".format(epoch=epoch, t=now)
    for k, v in errors.items():
        message = message + ",{name}:{err}".format(name=k, err=v)

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")
    return message


def print_args(config):
    print("======= Options ========")
    edict_type = type(edict())
    for k, v in sorted(config.items()):
        if type(v) == edict_type:
            print(k + ':')
            for k2, v2 in v.items():
                print("     {}: {}".format(k2, v2))
        else:
            print("{}: {}".format(k, v))
    print("========================")


def save_args(config_file_path, save_folder, save_name):
    if os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name)
    from shutil import copyfile
    copyfile(config_file_path, save_path)


def create_logger(cfg, cfg_name, phase='train'):
    this_dir = Path(os.path.dirname(__file__))
    root_dir = (this_dir / '..').resolve()
    output_dir = (root_dir / cfg.output_dir).resolve()
    log_dir = (root_dir / cfg.log_dir).resolve()
    if not output_dir.exists():
        print('Creating output dir {}'.format(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
    if not log_dir.exists():
        print('Creating log dir {}'.format(log_dir))
        log_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.dataset.name
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    # if cfg.dataset.category is not None:
    #     cfg_name += '_' + cfg.dataset.category
    exp_name = cfg.exp_name

    final_output_dir = output_dir / dataset_name / cfg_name / exp_name
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=str(final_log_file),
                        format=head,
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    #from IPython import embed; embed()

    tb_log_dir = log_dir / dataset_name / cfg_name / (exp_name + time_str)
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tb_log_dir)


def load_checkpoint(model, optimizer, resume_root, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    resume_path = os.path.join(resume_root, cpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = {"module.{}".format(key): item for key, item in checkpoint["state_dict"].items()}
        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))


def resume_training(model, optimizer, output_dir, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    output_dir = os.path.join(output_dir, cpt_name)
    if os.path.isfile(output_dir):
        print("=> loading checkpoint {}".format(output_dir))
        if device is not None:
            checkpoint = torch.load(output_dir, map_location=device)
        else:
            checkpoint = torch.load(output_dir, map_location=torch.device('cpu'))
        
        # load model
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)

        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])

        # load epoch
        start_epoch = checkpoint['epoch']

        best_psnr = checkpoint['best_psnr'] if 'best_psnr' in checkpoint.keys() else 0.0
        best_rot = checkpoint['best_rot'] if 'best_rot' in checkpoint.keys() else float('inf')
        
        return model, optimizer, start_epoch, best_psnr, best_rot
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(output_dir))


def load_encoder_pretrained(model, resume_root, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    resume_path = os.path.join(resume_root, cpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]

        # load model except for encoder_traj
        encoder_3d_dict = {key.replace('encoder_3d.',''): item for key, item in checkpoint["state_dict"].items() if 'encoder_3d' in key}
        num_missing_keys_encoder3d = len(encoder_3d_dict.keys()) - len(model.encoder_3d.state_dict().keys())
        if num_missing_keys_encoder3d > 0:
            warnings.warn("Encoder3D missing {} keys ! Please check the checkpoint.".format(num_missing_keys_encoder3d))

        rotate_dict = {key.replace('rotate.',''): item for key, item in checkpoint["state_dict"].items() if 'rotate' in key}
        num_missing_keys_rotate = len(rotate_dict.keys()) - len(model.rotate.state_dict().keys())
        if num_missing_keys_rotate > 0:
            warnings.warn("Rotate missing {} keys ! Please check the checkpoint.".format(num_missing_keys_rotate))

        render_dict = {key.replace('render.',''): item for key, item in checkpoint["state_dict"].items() if 'render' in key}
        num_missing_keys_render = len(render_dict.keys()) - len(model.render.state_dict().keys())
        if num_missing_keys_render > 0:
            warnings.warn("Render missing {} keys ! Please check the checkpoint.".format(num_missing_keys_render))
        
        model.rotate.load_state_dict(rotate_dict, strict=strict)
        model.encoder_3d.load_state_dict(encoder_3d_dict, strict=strict)
        model.render.load_state_dict(render_dict, strict=strict)
        print('Load pre-trained encoder_3d, rotate, render succesfully!')
        return model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))


def load_encoder_pretrained_pose(model, resume_root, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    resume_path = os.path.join(resume_root, cpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        
        num_missing_keys = len(model.state_dict().keys()) - len(model.encoder_traj.state_dict().keys()) - len(state_dict.keys())
        if num_missing_keys > 0:
            warnings.warn("Missing {} keys ! Please check the checkpoint.".format(num_missing_keys))

        # load model except for encoder_traj
        encoder_3d_dict = {key.replace('encoder_3d.',''): item for key, item in checkpoint["state_dict"].items() if 'encoder_3d' in key}
        num_missing_keys_encoder3d = len(encoder_3d_dict.keys()) - len(model.encoder_3d.state_dict().keys())
        if num_missing_keys_encoder3d > 0:
            warnings.warn("Encoder3D missing {} keys ! Please check the checkpoint.".format(num_missing_keys_encoder3d))
            
        model.encoder_3d.load_state_dict(encoder_3d_dict, strict=strict)
        print('Load pre-trained encoder_3d, rotate, render succesfully!')
        return model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))


def load_pose2d(model, resume_root, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    resume_path = os.path.join(resume_root, cpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]

        num_missing_keys = len(state_dict.keys()) - len(model.encoder_traj_2d.state_dict().keys())
        if num_missing_keys > 0:
            warnings.warn("EncoderTraj2D missing {} keys ! Please check the checkpoint.".format(num_missing_keys))
        model.encoder_traj_2d.load_state_dict(state_dict, strict=strict)

        print('Load pre-trained encoder_traj_2d succesfully!')
        return model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))
    

def load_pose3d(model, resume_root, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    resume_path = os.path.join(resume_root, cpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]

        encoder_traj_dict = {key.replace('encoder_traj.',''): item for key, item in checkpoint["state_dict"].items() if 'encoder_traj' in key}
        
        num_missing_keys_traj = len(encoder_traj_dict.keys()) - len(model.encoder_traj.state_dict().keys())
        if num_missing_keys_traj > 0:
            warnings.warn("EncoderTraj missing {} keys ! Please check the checkpoint.".format(num_missing_keys_traj))
        
        model.encoder_traj.load_state_dict(encoder_traj_dict, strict=strict)

        print('Load pre-trained encoder_traj succesfully!')
        return model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))


def load_model_finetune(model, resume_root, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    resume_path = os.path.join(resume_root, cpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        
        # num_missing_keys = len(model.state_dict().keys()) - len(state_dict.keys())
        # if num_missing_keys > 0:
        #     warnings.warn("Missing {} keys ! Please check the checkpoint.".format(num_missing_keys))

        # load model
        encoder_3d_dict = {key.replace('encoder_3d.',''): item for key, item in checkpoint["state_dict"].items() if 'encoder_3d' in key}
        num_missing_keys_encoder3d = len(encoder_3d_dict.keys()) - len(model.encoder_3d.state_dict().keys())
        if num_missing_keys_encoder3d > 0:
            warnings.warn("Encoder3D missing {} keys ! Please check the checkpoint.".format(num_missing_keys_encoder3d))

        rotate_dict = {key.replace('rotate.',''): item for key, item in checkpoint["state_dict"].items() if 'rotate' in key}
        num_missing_keys_rotate = len(rotate_dict.keys()) - len(model.rotate.state_dict().keys())
        if num_missing_keys_rotate > 0:
            warnings.warn("Rotate missing {} keys ! Please check the checkpoint.".format(num_missing_keys_rotate))

        render_dict = {key.replace('render.',''): item for key, item in checkpoint["state_dict"].items() if 'render' in key}
        num_missing_keys_render = len(render_dict.keys()) - len(model.render.state_dict().keys())
        if num_missing_keys_render > 0:
            warnings.warn("Render missing {} keys ! Please check the checkpoint.".format(num_missing_keys_render))

        encoder_traj_dict = {key.replace('encoder_traj.',''): item for key, item in checkpoint["state_dict"].items() if 'encoder_traj' in key}
        num_missing_keys_traj = len(encoder_traj_dict.keys()) - len(model.encoder_traj.state_dict().keys())
        if num_missing_keys_traj > 0:
            warnings.warn("EncoderTraj missing {} keys ! Please check the checkpoint.".format(num_missing_keys_render))
        
        model.rotate.load_state_dict(rotate_dict, strict=strict)
        model.encoder_3d.load_state_dict(encoder_3d_dict, strict=strict)
        model.render.load_state_dict(render_dict, strict=strict)
        model.encoder_traj.load_state_dict(encoder_traj_dict, strict=strict)
        print('Load Full pre-trained model (without 2D pose estimator) succesfully!')
        return model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))


def load_model_full(model, resume_root, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    resume_path = os.path.join(resume_root, cpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        
        num_missing_keys = len(model.state_dict().keys()) - len(state_dict.keys())
        if num_missing_keys > 0:
            warnings.warn("Missing {} keys ! Please check the checkpoint.".format(num_missing_keys))

        model.load_state_dict(state_dict, strict=strict)
        print('Load Full pre-trained model (including pose2d model) succesfully!')
        return model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))


def load_model_without_fusion(model, resume_root, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    resume_path = os.path.join(resume_root, cpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]

        fusion_keys = []
        for k in list(state_dict.keys()):
            if 'fusion_feature' in k:
                fusion_keys.append(k)
                del state_dict[k]
        #print(fusion_keys)

        model.load_state_dict(state_dict, strict=False)
        print('Load pre-trained encoder_3d, rotate, render, encoder_traj succesfully (without encoder_3d.fusion)!')
        return model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))


def load_pretrained_fusion(model, resume_root, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    resume_path = os.path.join(resume_root, cpt_name)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint {}".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]

        encoder_3d_dict = {key.replace('encoder_3d.fusion_feature.',''): item for key, item in checkpoint["state_dict"].items() if 'encoder_3d.fusion_feature.' in key}

        model.encoder_3d.fusion_feature.load_state_dict(encoder_3d_dict, strict=strict)
        print('Load pre-trained fusion!')
        return model
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))


def load_singleview_pretrain(config, encoder_3d, render):
    pretrain_file = config.train.sv_pretrain
    if len(pretrain_file) != 0 and os.path.isfile(pretrain_file):
        checkpoint = torch.load(pretrain_file, map_location='cpu')
        encoder_dict = checkpoint['encoder_3d_dict']
        render_dict = checkpoint['render_dict']
        if 'module' in list(encoder_dict.keys())[0]:
            encoder_dict = {key.replace('module.', ''): item for key, item in encoder_dict.items()}
        if 'module' in list(render_dict.keys())[0]:
            render_dict = {key.replace('module.', ''): item for key, item in render_dict.items()}
        encoder_3d.load_state_dict(encoder_dict, strict=True)
        render.load_state_dict(render_dict, strict=True)

        del checkpoint, encoder_dict, render_dict
    return encoder_3d, render


def save_checkpoint(state, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

