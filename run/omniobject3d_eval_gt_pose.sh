#!/bin/bash
cd "$(../ "$0")"

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --master_port 1234 --nproc_per_node=1 \
omniobject_train_pose_3D.py --cfg ./config/omniobject3d/gt_pose.yaml