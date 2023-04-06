# FORGE: Few-View Object Reconstruction with Unknown Categories and Camera Poses

### [Project Page](https://ut-austin-rpl.github.io/FORGE/) |  [Paper](https://arxiv.org/pdf/2212.04492.pdf)
<br/>

> Few-View Object Reconstruction with Unknown Categories and Camera Poses  
> [Hanwen Jiang](https://hwjiang1510.github.io/), [Zhenyu Jiang](https://zhenyujiang.me/), [Kristen Grauman](https://www.cs.utexas.edu/users/grauman/), [Yuke Zhu](https://cs.utexas.edu/~yukez)

![demo_vid](assets/forge-github-demo.gif)

## Installation

```
conda create --name forge python=3.8
conda activate forge

# Install pytorch or use your own torch version
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install pytorch3d, please follow https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# We use pytorch3d-0.7.0-py38_cu113_pyt1100

pip install -r requirements.txt
```


## Run FORGE demo
- Download pretrained weights for both [model A]() and [model B]().
- Put them in `./output/kubric/gt_pose/gt_pose/` and `./output/kubric/joint_pose_2d3d/pred_pose_2d3d_joint_train/`, respectively.
- Run demo on real images with `python demo.py --cfg ./config/demo/demo.yaml`.


## Train FORGE

### Download Dataset

- You can download our datasets created by Kubric. We train and test on the [shapenet-based-dataset](). We also test on the [Google-scanned-object-dataset](). Modify `self.root` in ./dataset/kubric.py and ./dataset/gso.py

### Train FORGE
1. Train the model using single 3D-based pose estimator.
    - Train the model using GT poses, using `./run/kubric_train_pose_3D_gt_pose.sh`.
    - Train the 3D-based pose estimator, using `./run/kubric_train_pose_3D_pred_pose.sh`.
    - (Optionally) Tune the model jointly, using `./run/kubric_train_pose_3D_pred_pose_joint.sh`.
2. Train the 2D-based pose estimator, using `./run/kubric_train_pose_2D.sh`.
3. Merge two pose estimator, finetune the model jointly.
    - Warmup training for two pose estimators, using `./run/kubric_train_pose_2D3D_head.sh`.
    - Train the whole pose estimator, using `./run/kubric_train_pose_2D3D.sh`.
    - Finetune the whole model, using `./run/kubric_train_pose_2D3D_finetune.sh`.


## Evaluate
We compare results with and without test-time optimization.
- For ShapeNet seen categories, using `./run/kubric_eval_seen.sh`.
- For ShapeNet unseen categories, using `./run/kubric_eval_unseen.sh`.
- For GSO unseen categories, using `./run/gso_eval.sh`.

The visualization and evaluation logs are saved in corresponding path specified by the configs. Use `./run/readout_eval.sh` to readout results.

You can try to use camera synchronization by adding argument `--use_sync` (default unused, it collapses under large pose errors).

## TODO
- [ ] Run FORGE on CO3D.

## Citation
```bibtex
@article{jiang2022forge,
   title={Few-View Object Reconstruction with Unknown Categories and Camera Poses},
   author={Jiang, Hanwen and Jiang, Zhenyu and Grauman, Kristen and Zhu, Yuke},
   journal={ArXiv},
   year={2022},
   volume={2212.04492}
}
```


