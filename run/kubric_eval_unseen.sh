CUDA_VISIBLE_DEVICES=0 python kubric_eval.py --split_num 8 --exp_id 0 --cfg ./config/kubric/optimize_unseen.yaml --model_gt &
CUDA_VISIBLE_DEVICES=1 python kubric_eval.py --split_num 8 --exp_id 1 --cfg ./config/kubric/optimize_unseen.yaml --model_gt &
CUDA_VISIBLE_DEVICES=2 python kubric_eval.py --split_num 8 --exp_id 2 --cfg ./config/kubric/optimize_unseen.yaml --model_gt &
CUDA_VISIBLE_DEVICES=3 python kubric_eval.py --split_num 8 --exp_id 3 --cfg ./config/kubric/optimize_unseen.yaml --model_gt &
CUDA_VISIBLE_DEVICES=4 python kubric_eval.py --split_num 8 --exp_id 4 --cfg ./config/kubric/optimize_unseen.yaml --model_gt &
CUDA_VISIBLE_DEVICES=5 python kubric_eval.py --split_num 8 --exp_id 5 --cfg ./config/kubric/optimize_unseen.yaml --model_gt &
CUDA_VISIBLE_DEVICES=6 python kubric_eval.py --split_num 8 --exp_id 6 --cfg ./config/kubric/optimize_unseen.yaml --model_gt &
CUDA_VISIBLE_DEVICES=7 python kubric_eval.py --split_num 8 --exp_id 7 --cfg ./config/kubric/optimize_unseen.yaml --model_gt