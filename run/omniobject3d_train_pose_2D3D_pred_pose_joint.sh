#!/bin/bash
cd "$(../ "$0")"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=8 \
omniobject_train_joint.py --cfg ./config/omniobject3d/joint_pose_2d3d.yaml