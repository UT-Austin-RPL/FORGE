CUDA_VISIBLE_DEVICES=0 python omniobject_eval.py --split_num 16 --exp_id 8 --cfg ./config/omniobject3d/optimize.yaml --model_gt &
CUDA_VISIBLE_DEVICES=1 python omniobject_eval.py --split_num 16 --exp_id 9 --cfg ./config/omniobject3d/optimize.yaml --model_gt &
CUDA_VISIBLE_DEVICES=2 python omniobject_eval.py --split_num 16 --exp_id 10 --cfg ./config/omniobject3d/optimize.yaml --model_gt &
CUDA_VISIBLE_DEVICES=3 python omniobject_eval.py --split_num 16 --exp_id 11 --cfg ./config/omniobject3d/optimize.yaml --model_gt &
CUDA_VISIBLE_DEVICES=4 python omniobject_eval.py --split_num 16 --exp_id 12 --cfg ./config/omniobject3d/optimize.yaml --model_gt &
CUDA_VISIBLE_DEVICES=5 python omniobject_eval.py --split_num 16 --exp_id 13 --cfg ./config/omniobject3d/optimize.yaml --model_gt &
CUDA_VISIBLE_DEVICES=6 python omniobject_eval.py --split_num 16 --exp_id 14 --cfg ./config/omniobject3d/optimize.yaml --model_gt &
CUDA_VISIBLE_DEVICES=7 python omniobject_eval.py --split_num 16 --exp_id 15 --cfg ./config/omniobject3d/optimize.yaml --model_gt