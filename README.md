# FORGE: Few-View Object Reconstruction with Unknown Categories and Camera Poses

### [Project Page](https://ut-austin-rpl.github.io/FORGE/) |  [Paper](https://arxiv.org/pdf/2212.04492.pdf)

<br/>

```
> Few-View Object Reconstruction with Unknown Categories and Camera Poses  
> [Hanwen Jiang](https://hwjiang1510.github.io/), [Zhenyu Jiang](https://zhenyujiang.me/), [Kristen Grauman](https://www.cs.utexas.edu/users/grauman/), [Yuke Zhu](https://cs.utexas.edu/~yukez)
```



![demo_vid](assets/forge-github-demo.gif)



## Installation

```
​```
conda create --name forge python=3.8
conda activate forge

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge	# or use your own torch version

# follow https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md to install PyTorch3D. We use pytorch3d-0.7.0-py38_cu113_pyt1100

pip install -r requirements.txt
​```
```



## Run FORGE demo

### Download FORGE model weights

- Download pretrained weights for both [model A]() and [model B]().
- Put them in 
- Run demo on real images with

```
​```
python demo.py --cfg ./config/demo/demo.yaml
​```
```



## Train FORGE

### Download Dataset

- You can download our datasets created by Kubric. We train and test on the [shapenet-based-dataset](). We also test on the [Google-scanned-object-dataset]().
- Modify self.root in ./dataset/kubric.py and ./dataset/gso.py



### Train FORGE

- Train the model with 3D-based pose estimator
- Train the 2D-based pose estimator
- Tune the model jointly



## Evaluate



