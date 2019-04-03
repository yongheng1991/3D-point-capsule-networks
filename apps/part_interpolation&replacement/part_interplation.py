#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:45:51 2018

@author: zhao
"""

import argparse
import torch
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../dataloaders')))
import shapenet_part_loader
import matplotlib.pyplot as plt

from pointcapsnet_ae import PointCapsNet,PointCapsNetDecoder
from capsule_seg_net import CapsSegNet

import json
from open3d import *

def main():
    blue = lambda x:'\033[94m' + x + '\033[0m'
    cat_no={'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 
            'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 
            'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}    
    
#generate part label one-hot correspondence from the catagory:
    dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../../dataset'))
    oid2cpid_file_name=os.path.join(dataset_main_path, opt.dataset,'shapenetcore_partanno_segmentation_benchmark_v0/shapenet_part_overallid_to_catid_partid.json')        
    oid2cpid = json.load(open(oid2cpid_file_name, 'r'))   
    object2setofoid = {}
    for idx in range(len(oid2cpid)):
        objid, pid = oid2cpid[idx]
        if not objid in object2setofoid.keys():
            object2setofoid[objid] = []
        object2setofoid[objid].append(idx)
    
    all_obj_cat_file = os.path.join(dataset_main_path, opt.dataset, 'shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt')
    fin = open(all_obj_cat_file, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    objcats = [line.split()[1] for line in lines]
#    objnames = [line.split()[0] for line in lines]
#    on2oid = {objcats[i]:i for i in range(len(objcats))}
    fin.close()


    colors = plt.cm.tab10((np.arange(10)).astype(int))
    blue = lambda x:'\033[94m' + x + '\033[0m'

# load the model for point cpas auto encoder    
    capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_vec_size, opt.num_points)
    if opt.model != '':
        capsule_net.load_state_dict(torch.load(opt.model))
    if USE_CUDA:
        capsule_net = torch.nn.DataParallel(capsule_net).cuda()
    capsule_net=capsule_net.eval()

# load the model for only decoding
    capsule_net_decoder = PointCapsNetDecoder(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_vec_size, opt.num_points)
    if opt.model != '':
        capsule_net_decoder.load_state_dict(torch.load(opt.model),strict=False)  
    if USE_CUDA:
        capsule_net_decoder = capsule_net_decoder.cuda()
    capsule_net_decoder=capsule_net_decoder.eval()

 
    
# load the model for capsule wised part segmentation      
    caps_seg_net = CapsSegNet(latent_caps_size=opt.latent_caps_size, latent_vec_size=opt.latent_vec_size , num_classes=opt.n_classes)    
    if opt.part_model != '':
        caps_seg_net.load_state_dict(torch.load(opt.part_model))
    if USE_CUDA:
        caps_seg_net = caps_seg_net.cuda()
    caps_seg_net = caps_seg_net.eval()    
    

    train_dataset = shapenet_part_loader.PartDataset(classification=False, class_choice=opt.class_choice, npoints=opt.num_points, split='test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)        


    # init the list for point cloud to save the source and target model
    pcd_list_source=[]
    for i in range(opt.latent_caps_size):
        pcd = PointCloud()    
        pcd_list_source.append(pcd)        
    pcd_list_target=[]
    for i in range(opt.latent_caps_size):
        pcd = PointCloud()    
        pcd_list_target.append(pcd)   

    # init the list for point cloud to save the interpolated model
    inter_models_number=5# the number of interpolated models
    part_inter_no=1 # the part that is interpolated
    pcd_list_inter=[]
    for i in range(inter_models_number):
        pcd_list=[]
        for j in range(opt.latent_caps_size):
            pcd = PointCloud() 
            pcd_list.append(pcd)
        pcd_list_inter.append(pcd_list)


# apply a transformation in order to get a better view point
##airplane
    rotation_angle=np.pi/2
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    flip_transforms = [[1, 0, 0,-2],[0,cosval, -sinval,1.5],[0,sinval, cosval, 0],[0, 0, 0, 1]]
    flip_transforms_r = [[1, 0, 0,2],[0, 1, 0,-1.5],[0, 0, 1,0],[0, 0, 0, 1]]   
    flip_transformt = [[1, 0, 0,2],[0,cosval, -sinval,1.5],[0,sinval, cosval, 0],[0, 0, 0, 1]]
    flip_transformt_r  = [[1, 0, 0,-2],[0, 1, 0,-1.5],[0, 0, 1,0],[0, 0, 0, 1]]
    transform_range=np.arange(-4,4.5,2)


# init open3d visualizer
#    vis = Visualizer()
#    vis.create_window()
    colors = plt.cm.tab20((np.arange(20)).astype(int))


    for batch_id, data in enumerate(train_dataloader):

        points, part_label, cls_label= data        
        if not (opt.class_choice==None ):
            cls_label[:]= cat_no[opt.class_choice]
    
        if(points.size(0)<opt.batch_size):
            break
        
        target = cls_label            
        points_ = Variable(points)
        points_ = points_.transpose(2, 1)
        if USE_CUDA:
            points_ = points_.cuda()
        latent_caps, reconstructions= capsule_net(points_)
        reconstructions=reconstructions.transpose(1,2).data.cpu()
        
        
        # create one hot label for the current class
        cur_label_one_hot = np.zeros((2, 16), dtype=np.float32)
        for i in range(2):
            cur_label_one_hot[i, cls_label[i]] = 1
        cur_label_one_hot=torch.from_numpy(cur_label_one_hot).float()        
        expand =cur_label_one_hot.unsqueeze(2).expand(2,16,opt.latent_caps_size).transpose(1,2)
        
         # recontstruct from the original presaved capsule
        latent_caps, target,expand = Variable(latent_caps), Variable(target), Variable(expand)
        latent_caps,target,expand = latent_caps.cuda(), target.cuda(), expand.cuda()

        # predidt the part label of each capsule
        latent_caps_with_one_hot=torch.cat((latent_caps,expand),2)
        latent_caps_with_one_hot,expand=Variable(latent_caps_with_one_hot),Variable(expand)
        latent_caps_with_one_hot,expand=latent_caps_with_one_hot.cuda(),expand.cuda()
        latent_caps_with_one_hot=latent_caps_with_one_hot.transpose(2, 1)
        output_digit=caps_seg_net(latent_caps_with_one_hot)
        for i in range (2):
            iou_oids = object2setofoid[objcats[cls_label[i]]]
            non_cat_labels = list(set(np.arange(50)).difference(set(iou_oids)))
            mini = torch.min(output_digit[i,:,:])
            output_digit[i,:, non_cat_labels] = mini - 1000
        pred_choice = output_digit.data.cpu().max(2)[1]        
        
        
        # Append the capsules which is used to reconstruct the same part in two shapes
        part_no=iou_oids[part_inter_no]# translate part label form [0,4] to [0,50]
        part_viz=[]
        for caps_no in range (opt.latent_caps_size):
            if(pred_choice[0,caps_no]==part_no and pred_choice[1,caps_no]==part_no):
                part_viz.append(caps_no)
        
     
        # add source reconstruction to open3d point cloud container to viz
        for j in range(opt.latent_caps_size):
            current_patch=torch.zeros(int(opt.num_points/opt.latent_caps_size),3)
            for m in range(int(opt.num_points/opt.latent_caps_size)):
                current_patch[m,]=reconstructions[1][opt.latent_caps_size*m+j,]      
            pcd_list_source[j].points = Vector3dVector(current_patch)
            if(j in part_viz):
                pcd_list_source[j].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
            else:
                pcd_list_source[j].paint_uniform_color([0.8,0.8,0.8])


        # add target reconstruction to open3d point cloud container to viz
        for j in range(opt.latent_caps_size):
            current_patch=torch.zeros(int(opt.num_points/opt.latent_caps_size),3)
            for m in range(int(opt.num_points/opt.latent_caps_size)):
                current_patch[m,]=reconstructions[0][opt.latent_caps_size*m+j,]             
            pcd_list_target[j].points = Vector3dVector(current_patch)
            if(j in part_viz):
                pcd_list_target[j].paint_uniform_color([colors[5,0], colors[5,1], colors[5,2]])
            else:
                pcd_list_target[j].paint_uniform_color([0.8,0.8,0.8])

        
        # interpolate the latent capsules between two shape
        latent_caps_inter=torch.zeros(inter_models_number,opt.latent_caps_size,opt.latent_vec_size)  
        latent_caps_inter= Variable(latent_caps_inter)
        latent_caps_inter = latent_caps_inter.cuda()
        
        latent_caps_st_diff=torch.zeros(len(part_viz),opt.latent_caps_size)
        latent_caps_st_diff=latent_caps_st_diff.cuda()
        for j in range (len(part_viz)):
            latent_caps_st_diff[j,]=latent_caps[0,part_viz[j],]-latent_caps[1,part_viz[j],]               
 
        for i in range (inter_models_number):
            latent_caps_inter[i,]=latent_caps[1,]
            for j in range (len(part_viz)):
                latent_caps_inter[i,part_viz[j],]=latent_caps[1,part_viz[j],] + latent_caps_st_diff[j,] * i  / (inter_models_number-1)

        # decode the interpolated latent capsules
        reconstructions_inter = capsule_net_decoder(latent_caps_inter)
        reconstructions_inter=reconstructions_inter.transpose(1,2).data.cpu()
       
        # add interpolated latent capsule reconstruction to open3d point cloud container to viz
        for i in range (inter_models_number):
            for j in range(opt.latent_caps_size):
                current_patch=torch.zeros(int(opt.num_points/opt.latent_caps_size),3)
                for m in range(int(opt.num_points/opt.latent_caps_size)):
                    current_patch[m,]=reconstructions_inter[i][opt.latent_caps_size*m+j,]             
                pcd_list_inter[i][j].points = Vector3dVector(current_patch)
                part_no=iou_oids[part_inter_no]                      
                if(j in part_viz):
                    pcd_list_inter[i][j].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
                else:
                    pcd_list_inter[i][j].paint_uniform_color([0.8,0.8,0.8])

        
        # show all interpolation
        all_point=PointCloud()
        for j in range (opt.latent_caps_size):
            pcd_list_source[j].transform(flip_transforms)
            pcd_list_target[j].transform(flip_transformt)
            all_point+=pcd_list_source[j]
            all_point+=pcd_list_target[j]
                
        for r in range(inter_models_number):
            flip_transform_inter  = [[1, 0, 0,transform_range[r]],[0,cosval, -sinval,-2],[0,sinval, cosval, 0],[0, 0, 0, 1]]
            for k in range (opt.latent_caps_size):
                pcd_list_inter[r][k].transform(flip_transform_inter)
                all_point+=pcd_list_inter[r][k]
        draw_geometries([all_point])
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--part_model', type=str, default='../../checkpoints/part_seg_100percent.pth', help='model path for the pre-trained part segmentation network')
    parser.add_argument('--model', type=str, default='../../checkpoints/shapenet_part_dataset_ae_200.pth', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55, modelent40')
    parser.add_argument('--n_classes', type=int, default=50, help='part classes in all the catagories')
    parser.add_argument('--class_choice', type=str, default='Airplane', help='choose the class to eva')

    opt = parser.parse_args()
    print(opt)

    USE_CUDA = True
    main()
    
  


#
