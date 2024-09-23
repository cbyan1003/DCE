# Discriminative Correspondence Estimation for Unsupervised RGB-D Point Cloud Registration

## Instructions
This code has been tested on 
- Python 3.8, PyTorch 1.12.1, CUDA 11.3, GeForce RTX 3090

## Requirements
To create a virtual environment and install the required dependences please run:
```shell
git clone https://github.com/cbyan1003/DCE.git
conda create --name DCE python=3.8
conda activate DCE
pip install -r requirements.txt
```

## Train & Test

### Train on 3DMatch
```shell
python train.py --name RGBD_3DMatch  --RGBD_3D_ROOT 
```

### Train on ScanNet
```shell
python train.py --name ScanNet  --SCANNET_ROOT 
```

### Inference
```shell
python test.py --checkpoint --SCANNET_ROOT
```
## CheckPoint
[3DMatch.pkl](https://pan.baidu.com/s/1FZlxfU6oaCXiVdsiwuE9FA?pwd=fcwy "3DMatch")

## Visualization Samples

![Visualization in scannet](https://github.com/user-attachments/assets/f049ad36-acb4-466d-a728-013af718c5a0)

![Visualization in 3dmatch](https://github.com/user-attachments/assets/f288d53b-3a21-4bde-a38e-3d046ac019a4)
