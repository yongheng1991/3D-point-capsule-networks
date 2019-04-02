## 3D-point-capsule-networks
Created by <a href="http://campar.in.tum.de/Main/YongHengZhao" target="_blank">Yongheng Zhao</a>, <a href="http://campar.in.tum.de/Main/TolgaBirdal" target="_blank">Tolga Birdal</a>, <a href="http://campar.in.tum.de/Main/HaowenDeng" target="_blank">Haowen Deng</a>, <a href="http://campar.in.tum.de/Main/FedericoTombari" target="_blank">Federico Tombari </a> from TUM.


![](https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/teaser.png )

##### Part Interpolation 
<a href="url"><img src="https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/airplane_wing_interpolation.gif"  height="210" width="260" ></a>
<a href="url"><img src="https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/chair2.gif" height="220" width="145" ></a>
<a href="url"><img src="https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/tableleg2.gif" height="110" width="228" ></a>
<a href="url"><img src="https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/table_surface2.gif" height="173" width="179" ></a>

##### Distribution of capsule reconstruction 
![](https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/spatial_attention.png )

### Introduction
This work is based on our [arXiv tech report](https://arxiv.org/abs/1812.10775), which is going to appear in CVPR 2019.

### Citation
If you find our work useful in your research, please consider citing:

          @article{zhao20183d,
            title={3D Point-Capsule Networks},
            author={Zhao, Yongheng and Birdal, Tolga and Deng, Haowen and Tombari, Federico},
            journal={arXiv preprint arXiv:1812.10775},
            year={2018}
          }
   

### Installation

The code is based on Pytorch. It has been tested with Python 3.6, Pytorch 1.0.0, CUDA 9.2 on Ubuntu 16.04.

To viz the training process in pytorch:
Install  <a href="https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard" target="_blank">Tensorboard</a>


To viz the reconstucted point cloud:
Install <a href="http://www.open3d.org/docs/getting_started.html" target="_blank">Open3D</a> 


Install h5py for Python:
```bash
  sudo apt-get install libhdf5-dev
  sudo pip install h5py
```

Install Chamfer distance package:
```bash
  cd models/nndistance
  python build.py install
```


### Dataset

#### ShapeNetPart dataset
```bash
  cd dataset
  bash download_shapenet_part16_catagories.sh
```
#### ShapeNet Core with 13 catagories (The same astlasnet)
```bash
  cd dataset
  bash download_shapenet_core13_catagories.sh
```
#### ShapeNet Core with 55 catagories (The same as Foldingnet)
```bash
  cd dataset
  bash download_shapenet_core55_catagories.sh
```


### Usage
#### Minimized Example

The minimized example is in folder 'mini_example':
To train a point capsule auto encoder with ShapeNetPart dataset:
```bash
  cd mini_example/AE
  python train_ae.py
```
To visualize the reconstruction from latent capsules:
```bash
  cd mini_example/AE
  python viz_reconstruction.py --model shapenet_part_dataset__64caps_64vec_70.pth
```

#### Point Capsule Auto Encoder

To train a point capsule auto encoder with other datasets:
```bash
  cd apps/AE
  python train_ae.py --dataset < shapenet_part, shapenet_core13, shapenet_core55 >
```

To view training process, use tensorboard with seting the log dir:
```bash
  tensorboard --logdir log
```
To test the reconstruction accuracy:
```bash
  python test_ae.py  --dataset < >  --model < >
```
To viz the reconstruction:
```bash
  python viz_reconstruction.py --dataset < >  --model < >
```

#### Transfer Classification and Semi Classification

To generate latent capsules from pre-trained model and save them as a file:
```bash
  cd apps/trasfer_cls
  python save_output_latent_caps_in_file.py --dataset < >  --model < >  --save_training  # process and save training dataset
  python save_output_latent_caps_in_file.py --dataset < >  --model < >   # process and save testing dataset
```
To train and test liner SVM cls with the pre-trained AE model and pre-saved latent capsules. The AE model and latent capsules are obtianed from different dataset in order to test the transfer classification.
```bash
  python train_and_test_svm_cls_from_pre-saved_latent_caps.py --dataset < >  --model < >
```

To train Liner SVM cls with part of training data and test with all the testing data:
```bash
  python train_and_test_svm_cls_from_pre-saved_latent_caps.py --dataset < >  --model < > --percent_training_dataset < 5, 10 ...>
```

#### Part segmentation
To generate latent capsules with the part label from pre-trained model and save them as a file (The model is trained also with shapenet-part dataset):
```bash
  cd apps/part_seg
  python save_output_latent_caps_with_part_label.py --dataset shapenet_part  --model < >  --save_training  # process and save training dataset
  python save_output_latent_caps_with_part_label.py --dataset shapenet_part  --model < >   # process and save testing dataset
```
To train a capsule wised part segmentation with a specific amout of training data:
```bash
  python train_seg.py --model < > --percent_training_dataset < 5, 10 ...>
 ```
 To evaluate and visualize the part segmentation:
 (--model < pre-trained model of point capsule auto encoder >;
 --part_model <pre-trained model of part segmentation >)
          
 ```bash
  python eva_seg.py --model < >  --part_model < > --class_choice Table
  ```

#### Part interpolation 
to be continued...


####
3D point match 
to be continued...


### License
Our code is released under MIT License (see LICENSE file for details).

### Reference 
The chamfer distance package is based on <a href="https://github.com/fxia22/pointGAN/tree/master/nndistance" target="_blank">nndistance</a>. The modification has been done in this depo in order to run it with Pytorch 1.0.0.

The capsule layer has refered and modified from  <a href="https://github.com/higgsfield/Capsule-Network-Tutorial" target="_blank">Capsule-Network-Tutorial</a>

The capsule decoder has refered the decoder form <a href="https://github.com/ThibaultGROUEIX/AtlasNet" target="_blank">Atlasnet</a>.


