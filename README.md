

## 3D Point Capsule Networks
Created by <a href="http://campar.in.tum.de/Main/YongHengZhao" target="_blank">Yongheng Zhao</a>, <a href="http://campar.in.tum.de/Main/TolgaBirdal" target="_blank">Tolga Birdal</a>, <a href="http://campar.in.tum.de/Main/HaowenDeng" target="_blank">Haowen Deng</a>, <a href="http://campar.in.tum.de/Main/FedericoTombari" target="_blank">Federico Tombari </a> from TUM.

This repository contains the implementation of our [CVPR 2019 paper *3D Point Capsule Networks*](https://arxiv.org/abs/1812.10775). In particular, we release code for training and testing a 3D-PointCapsNet network for classification, reconstruction, part interpolation and extraction of 3d local descriptors as well as the pre-trained models for quickly replicating our results. 

For an intuitive explanation of the 3D point capsule networks, please check out [Tolga's CVPR tutorial](https://www.youtube.com/watch?v=fbhbuH9mUx0).

![](https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/teaser.png )

##### Part Interpolation 
<a href="url"><img src="https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/airplane_wing_interpolation.gif"  height="210" width="260" ></a>
<a href="url"><img src="https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/chair2.gif" height="220" width="145" ></a>
<a href="url"><img src="https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/tableleg2.gif" height="110" width="228" ></a>
<a href="url"><img src="https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/table_surface2.gif" height="173" width="179" ></a>

##### Distribution of capsule reconstruction 
![](https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/docs/spatial_attention.png )

#### Abstract
In this paper, we propose 3D point capsule networks, an auto-encoder designed to process sparse 3D point clouds while preserving spatial arrangements of the input data. 3D capsule networks arise as a direct consequence of our novel unified 3D auto-encoder formulation. Their dynamic routing scheme and the peculiar 2D latent space deployed by our approach bring in improvements for several common point cloud-related tasks, such as object classification, object reconstruction and part segmentation as substantiated by our extensive evaluations. Moreover, it enables new applications such as part interpolation and replacement.

### Citation
If you find our work useful in your research, please consider citing:
		  
		  @inproceedings{zhao20193d, 
			author={Zhao, Yongheng and Birdal, Tolga and Deng, Haowen and Tombari, Federico}, 
			booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)}, 
			title={3D Point 
			Capsule Networks}, 
			organizer={IEEE/CVF},
			year={2019}
		  }
   

### Installation

The code is based on PyTorch. It has been tested with Python 3.6, PyTorch 1.0.0, CUDA 9.2(or higher) on Ubuntu 16.04.
(You can also use PyTorch 0.4.0 but you need to replace the Chamfer distance package with the original <a href="https://github.com/fxia22/pointGAN/tree/master/nndistance" target="_blank">nndistance</a>.)

Install h5py for Python:
```bash
  sudo apt-get install libhdf5-dev
  sudo pip install h5py
```

Install Chamfer Distance(CD) package:
(Be aware of the PyTorch 1.0.1. It may have a problem for building this cuda package.)
```bash
  cd models/nndistance
  python build.py install
```
(In case you are using pytorch version higher than 1.0, you could use the updated chamfer distance package named "torch-nndistance". But you need to modify the package usage in the several scripts in which the CD library is used. You can find "test.py" in the updated package folder for the usage reference.


To visualize the training process in PyTorch, consider installing  <a href="https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard" target="_blank">TensorBoard</a>.

To visualize the reconstructed point cloud, consider installing <a href="http://www.open3d.org/docs/getting_started.html" target="_blank">Open3D</a>.

### Datasets

#### ShapeNetPart Dataset
```bash
  cd dataset
  bash download_shapenet_part16_catagories.sh
```
#### ShapeNet Core with 13 categories (refered from <a href="https://github.com/ThibaultGROUEIX/AtlasNet" target="_blank">AtlasNet</a>.)
```bash
  cd dataset
  bash download_shapenet_core13_catagories.sh
```
#### ShapeNet Core with 55 categories (refered from <a href="http://www.merl.com/research/license#FoldingNet" target="_blank">FoldingNet</a>.)
```bash
  cd dataset
  bash download_shapenet_core55_catagories.sh
```

### Pre-trained model
You can download the pre-trained models <a href="https://drive.google.com/drive/folders/1XgYWPjAFgn4Vdzm3AjWnGJYFS6Ho9pm5?usp=sharing" target="_blank">here</a>.


### Usage
#### A Minimal Example

We provide an example demonstrating the basic usage in the folder 'mini_example'. 

To visualize the reconstruction from latent capsules with our pre-trained model:
```bash
  cd mini_example/AE
  python viz_reconstruction.py --model ../../checkpoints/shapenet_part_dataset_ae_200.pth
```

To train a point capsule auto encoder with ShapeNetPart dataset by yourself:
```bash
  cd mini_example/AE
  python train_ae.py
```
#### Point Capsule Auto Encoder

To train a point capsule auto encoder with another dataset:
```bash
  cd apps/AE
  python train_ae.py --dataset < shapenet_part, shapenet_core13, shapenet_core55 >
```

To monitor the training process, use TensorBoard by specifying the log directory:
```bash
  tensorboard --logdir log
```
To test the reconstruction accuracy:
```bash
  python test_ae.py  --dataset < >  --model < >
e.g. 
  python test_ae.py --dataset shapenet_core13 --model ../../checkpoints/shapenet_core13_dataset_ae_230.pth
```

To visualize the reconstructed points:
```bash
  python viz_reconstruction.py --dataset < >  --model < >
e.g. 
  python viz_reconstruction.py --dataset shapenet_core13 --model ../../checkpoints/shapenet_core13_dataset_ae_230.pth
```

#### Transfer Learning and Semi Supervised Classification

To generate latent capsules from a pre-trained model and save them into a file:
```bash
  cd apps/trasfer_cls
  python save_output_latent_caps_in_file.py --dataset < >  --model < >  --save_training  # process and save training dataset
  python save_output_latent_caps_in_file.py --dataset < >  --model < >   # process and save testing dataset
```
To train and test the liner SVM classifier with the pre-trained AE model and pre-saved latent capsules:
```bash
  python train_and_test_svm_cls_from_pre-saved_latent_caps.py --dataset < >  --model < >
```
The AE model and latent capsules are obtained from different datasets in order to test the performance of classification under transfer.

Training a Liner SVM classifier with a limited part of the training data and testing with the complete test data:
```bash
  python train_and_test_svm_cls_from_pre-saved_latent_caps.py --dataset < >  --model < > --percent_training_dataset < 5, 10 ...>
e.g.
  python train_and_test_svm_cls_from_pre-saved_latent_caps.py --dataset shapenet_part  --model ../../checkpoints/shapenet_part_dataset_ae_200.pth --percent_training_dataset 10
```

#### Part Segmentation
To generate latent capsules with the part label from a pre-trained model and save them into a file (The model is also trained with shapenet-part dataset):
```bash
  cd apps/part_seg
  python save_output_latent_caps_with_part_label.py --dataset shapenet_part  --model < >  --save_training  # process and save training dataset
  python save_output_latent_caps_with_part_label.py --dataset shapenet_part  --model < >   # process and save testing dataset
```
To train a capsule-wise part segmentation with a specific amount of training data:
```bash
  python train_seg.py --model < > --percent_training_dataset < 5, 10 ...>
e.g.
  python train_seg.py --model ../../checkpoints/shapenet_part_dataset_ae_200.pth --percent_training_dataset 1
 ```
To evaluate and visualize the part segmentation:
 (--model < pre-trained model of point capsule auto encoder >;
 --part_model <pre-trained model of part segmentation >)
          
 ```bash
  python eva_seg.py --model < >  --part_model < > --class_choice < >
e.g.
  python eva_seg.py --model ../../checkpoints/shapenet_part_dataset_ae_200.pth  --part_model ../../checkpoints/part_seg_1percent.pth  --class_choice Airplane
  ```

#### Part Interpolation and Replacement
To visualize the part interpolation in open3D:
 ```bash
  python part_interplation.py --model < >  --part_model < > --class_choice < >
e.g.
  python part_interplation.py --model ../../checkpoints/shapenet_part_dataset_ae_200.pth  --part_model ../../checkpoints/part_seg_100percent.pth  --class_choice Airplane
 ```

To visualize the part replacement in open3D:
 ```bash
  python part_replacement.py --model < >  --part_model < > --class_choice < >
 ```

#### 3D Local Feature Extraction
to be continued...


### License
Our code is released under MIT License (see LICENSE file for details).

### Reference 
The chamfer distance package is based on <a href="https://github.com/fxia22/pointGAN/tree/master/nndistance" target="_blank">nndistance</a>. The necessary modifications have been done to this repository in order to run it with PyTorch 1.0.0.

The capsule layer is based upon and modified from <a href="https://github.com/higgsfield/Capsule-Network-Tutorial" target="_blank">Capsule-Network-Tutorial</a>

Our capsule decoder is based upon the decoder of <a href="https://github.com/ThibaultGROUEIX/AtlasNet" target="_blank">AtlasNet</a>.
