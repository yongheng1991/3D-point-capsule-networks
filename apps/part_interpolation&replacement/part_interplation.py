#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:45:51 2018

@author: zhao
"""



import argparse
import copy
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import random
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
# from datasets_cls import PartDataset
import json
from collections import OrderedDict
sys.path.append('../models')
sys.path.append('../my_dataloader')
sys.path.append('../viz')
import shapepart_multi_loader
import modelnet_pointnet2_loader
import shapepart_loader
import modelnet_pointnet_loader

from pointcapsnet_ae import PointCapsNet,PointCapsNetDecoder
from semi_caps import SemiConvSegNet
from logger import Logger
from open3d import *
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'nndistance'))
from modules.nnd import NNDModule
distChamfer = NNDModule()


USE_CUDA = True
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--n_classes', type=int, default=50, help='number of classed')
parser.add_argument('--num_points', type=int, default=2048, help='number of poins')
#parser.add_argument('--ae_model', type=str,
#                    default='../checkpoints/atlas_ae_dataset_atlascaps_decoder_230.pth', help='model path')
parser.add_argument('--ae_model', type=str,
                    default='../checkpoints/shapenet_part_dataset_atlascaps_decoder_200.pth', help='model path')

parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset name')

parser.add_argument('--dataset_augmentation', type=bool, default=False,
                    help='if do the points augmentation')
parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')

parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of prim_caps')
parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of prim_vec')
parser.add_argument('--digit_caps_size', type=int, default=64, help='number of digit_caps')
parser.add_argument('--digit_vec_size', type=int, default=64, help='scale of digit_vec')

parser.add_argument('--decoder', type=str, default='ae1', help='ae ')
#parser.add_argument('--part_model', type=str, default='../checkpoints/seg_model_20.pth', help='model path')
parser.add_argument('--part_model', type=str, default='../checkpoints/part_seg_all_codeword_100percent_100.pth', help='model path')

cat_no={'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 
 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 
 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

parser.add_argument('--class_choice', type=str, default='Airplane', help='aclass_choice ')

part_inter_no=1
viz_p=False
save_p=True
use_threshold_in_acc=False
show_part_inter=True


opt = parser.parse_args()
print(opt)

oid2cpid = json.load(open(os.path.join('/home/zhao/Code/dataset/shapenet_part', 'overallid_to_catid_partid.json'), 'r'))

object2setofoid = {}
for idx in range(len(oid2cpid)):
    objid, pid = oid2cpid[idx]
    if not objid in object2setofoid.keys():
        object2setofoid[objid] = []
    object2setofoid[objid].append(idx)
all_obj_cat_file = os.path.join('/home/zhao/Code/dataset/shapenet_part', 'all_object_categories.txt')
fin = open(all_obj_cat_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
objcats = [line.split()[1] for line in lines]
objnames = [line.split()[0] for line in lines]
on2oid = {objcats[i]:i for i in range(len(objcats))}
fin.close()    
    

capsule_net_decoder = PointCapsNetDecoder(opt.prim_caps_size, opt.prim_vec_size, opt.digit_caps_size, opt.digit_vec_size, opt.num_points, ae=opt.decoder)

if opt.ae_model != '':
    capsule_net_decoder.load_state_dict(torch.load(opt.ae_model),strict=False)
    
if USE_CUDA:
    capsule_net_decoder = capsule_net_decoder.cuda()
capsule_net_decoder=capsule_net_decoder.eval()


capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.digit_caps_size, opt.digit_vec_size, opt.num_points, ae=opt.decoder)

if opt.ae_model != '':
    capsule_net.load_state_dict(torch.load(opt.ae_model))

if USE_CUDA:
    capsule_net = torch.nn.DataParallel(capsule_net).cuda()

capsule_net=capsule_net.eval()


semi_capsule_net = SemiConvSegNet(digit_caps_size=opt.digit_caps_size, digit_vec_size=opt.digit_vec_size , num_classes=opt.n_classes)
if USE_CUDA:
    semi_capsule_net = semi_capsule_net.cuda()
    
if opt.part_model != '':
    semi_capsule_net.load_state_dict(torch.load(opt.part_model))
semi_capsule_net=semi_capsule_net.eval()    


train_dataset = shapepart_loader.PartDataset(root='/home/zhao/Code/dataset/shapenet_part/',
                  classification=False, class_choice=opt.class_choice, npoints=opt.num_points, split='test')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                     shuffle=True, num_workers=4)


pcd_list_source=[]
for i in range(64):
    pcd = PointCloud()    
    pcd_list_source.append(pcd)
    
pcd_list_target=[]
for i in range(64):
    pcd = PointCloud()    
    pcd_list_target.append(pcd)    

    

inter_models_number=5
pcd_list_inter=[]
for i in range(inter_models_number):
    pcd_list=[]
    for j in range(64):
        pcd = PointCloud() 
        pcd_list.append(pcd)
    pcd_list_inter.append(pcd_list)



#chair
#rotation_angle=-np.pi/2
#cosval = np.cos(rotation_angle)
#sinval = np.sin(rotation_angle)
#flip_transforms  = [[cosval, 0, sinval,-2],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
#flip_transforms_r  = [[1, 0, 0, 2],[0, 1, 0,0],[0, 0, 1,0],[0, 0, 0, 1]]
#flip_transformt  = [[cosval, 0, sinval,2],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
#flip_transformt_r  = [[1, 0, 0, -2],[0, 1, 0,0],[0, 0, 1,0],[0, 0, 0, 1]]
#transform_range=np.arange(4.5,-4,-2)



##airplane
rotation_angle=np.pi/2
cosval = np.cos(rotation_angle)
sinval = np.sin(rotation_angle)
flip_transforms = [[1, 0, 0,-2],[0,cosval, -sinval,1.5],[0,sinval, cosval, 0],[0, 0, 0, 1]]
flip_transforms_r = [[1, 0, 0,2],[0, 1, 0,-1.5],[0, 0, 1,0],[0, 0, 0, 1]]

flip_transformt = [[1, 0, 0,2],[0,cosval, -sinval,1.5],[0,sinval, cosval, 0],[0, 0, 0, 1]]
flip_transformt_r  = [[1, 0, 0,-2],[0, 1, 0,-1.5],[0, 0, 1,0],[0, 0, 0, 1]]
transform_range=np.arange(-4,4.5,2)



if(viz_p):
    vis = Visualizer()
    vis.create_window()
colors = plt.cm.tab20((np.arange(20)).astype(int))
   


n_batch = 0
file_no=0
for epoch in range(100):
    for batch_id, data in enumerate(train_dataloader):
        points, part_label, cls_label,fn= data
        if(points.size(0)<opt.batch_size):  
            break
        print ('batch id', batch_id)
        if not (opt.class_choice==None):
            cls_label[:]= cat_no[opt.class_choice]
        if (batch_id!=-1): 
    #    if (batch_id==20):  
            
            target = cls_label
            
            points_ = Variable(points)
            points_ = points_.transpose(2, 1)
            if USE_CUDA:
                points_ = points_.cuda()
            capsmap, reconstructions= capsule_net(points_)
            reconstructions=reconstructions.transpose(1,2).data.cpu()
        
            cur_label_one_hot = np.zeros((2, 16), dtype=np.float32)
            for i in range(2):
                cur_label_one_hot[i, cls_label[i]] = 1
            cur_label_one_hot=torch.from_numpy(cur_label_one_hot).float()        
            expand =cur_label_one_hot.unsqueeze(2).expand(2,16,64).transpose(1,2)
        
    
            # recontstruct from the original presaved capsule
            capsmap, target,expand = Variable(capsmap), Variable(target), Variable(expand)
            capsmap,target,expand = capsmap.cuda(), target.cuda(), expand.cuda()
    
            # predidt the part label of each capsule
            capsmap_with_one_hot=torch.cat((capsmap,expand),2)
            capsmap_with_one_hot,expand=Variable(capsmap_with_one_hot),Variable(expand)
            capsmap_with_one_hot,expand=capsmap_with_one_hot.cuda(),expand.cuda()
            capsmap_with_one_hot=capsmap_with_one_hot.transpose(2, 1)
            
    
            output_digit=semi_capsule_net(capsmap_with_one_hot)
            for i in range (2):
                iou_oids = object2setofoid[objcats[cls_label[i]]]
                non_cat_labels = list(set(np.arange(50)).difference(set(iou_oids)))
                mini = torch.min(output_digit[i,:,:])
                output_digit[i,:, non_cat_labels] = mini - 1000
            pred_choice = output_digit.data.cpu().max(2)[1]
            
            
            
            if use_threshold_in_acc:
                # add a  filter to get better data
                for i in range(2):
                    iou_oids = object2setofoid[objcats[cls_label[i]]]
                    for j in range(2048):
                        part_label[i,j]=iou_oids[part_label[i,j]]
                reconstructions_part_label=torch.zeros([opt.batch_size,opt.num_points],dtype=torch.int64)
                if(opt.decoder=='ae1'):
                    for i in range(opt.batch_size):
                        for j in range(opt.digit_caps_size):
                            for m in range(int(opt.num_points/opt.digit_caps_size)):
                                reconstructions_part_label[i,64*m+j]=pred_choice[i,j]
                pcd=pcd = PointCloud() 
                pred_ori_pointcloud_part_label=torch.zeros([opt.batch_size,opt.num_points],dtype=torch.int64)
                for batch_no in range (opt.batch_size):
                    pcd.points = Vector3dVector(reconstructions[batch_no,])
                    pcd_tree = KDTreeFlann(pcd)
                    for point_id in range (opt.num_points):
                        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[batch_no,point_id,:], 1)
                        local_patch_labels=reconstructions_part_label[batch_no,idx]
                        pred_ori_pointcloud_part_label[batch_no,point_id]=local_patch_labels
                correct = pred_ori_pointcloud_part_label.eq(part_label.data.cpu()).cpu().sum()
                acc=correct.item()/float(opt.batch_size*opt.num_points)
                if(acc<0.9):
                    continue
            
        
            
            
            
            
            part_no=iou_oids[part_inter_no]
            part_viz=[]
            for caps_no in range (64):
                if(pred_choice[0,caps_no]==part_no and pred_choice[1,caps_no]==part_no):
                    part_viz.append(caps_no)
                        
                    
                    
            if show_part_inter:
                viz_caps_no=64
            else:
                viz_caps_no=20
             # add source reconstruction
            for j in range(viz_caps_no):
                current_patch=torch.zeros(32,3)
                for m in range(32):
                    current_patch[m,]=reconstructions[1][64*m+j,]      
                pcd_list_source[j].points = Vector3dVector(current_patch)
    
                if(j in part_viz):
                    pcd_list_source[j].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
                else:
                    pcd_list_source[j].paint_uniform_color([0.8,0.8,0.8])
                if not show_part_inter:
                    pcd_list_source[j].paint_uniform_color([colors[j,0], colors[j,1], colors[j,2]])    
                if(viz_p):
                    vis.add_geometry(pcd_list_source[j])
    
            # add target reconstruction
            for j in range(viz_caps_no):
                current_patch=torch.zeros(32,3)
                for m in range(32):
                    current_patch[m,]=reconstructions[0][64*m+j,]             
                pcd_list_target[j].points = Vector3dVector(current_patch)
                if(j in part_viz):
                    pcd_list_target[j].paint_uniform_color([colors[5,0], colors[5,1], colors[5,2]])
                else:
                    pcd_list_target[j].paint_uniform_color([0.8,0.8,0.8])
                if not show_part_inter:    
                    pcd_list_target[j].paint_uniform_color([colors[j,0], colors[j,1], colors[j,2]])
                if(viz_p):
                    vis.add_geometry(pcd_list_target[j])
    
         #   show part capsule colored based interpolation???????????????????????
                
            if show_part_inter:
                viz_caps_no=64
            else:
                viz_caps_no=20
            
            capsmap_inter=torch.zeros(inter_models_number,64,64)  
            capsmap_inter= Variable(capsmap_inter)
            capsmap_inter = capsmap_inter.cuda()
            
            capsmap_st_diff=torch.zeros(len(part_viz),64)
            capsmap_st_diff=capsmap_st_diff.cuda()
            for j in range (len(part_viz)):
                capsmap_st_diff[j,]=capsmap[0,part_viz[j],]-capsmap[1,part_viz[j],]               
            
            for i in range (inter_models_number):
                capsmap_inter[i,]=capsmap[1,]
                for j in range (len(part_viz)):
                    capsmap_inter[i,part_viz[j],]=capsmap[1,part_viz[j],] + capsmap_st_diff[j,] * i  / (inter_models_number-1)
    
            
            reconstructions_inter = capsule_net_decoder(capsmap_inter, target)
            reconstructions_inter=reconstructions_inter.transpose(1,2).data.cpu()
            for i in range (inter_models_number):
                for j in range(viz_caps_no):
                    current_patch=torch.zeros(32,3)
                    for m in range(32):
                        current_patch[m,]=reconstructions_inter[i][64*m+j,]             
                    pcd_list_inter[i][j].points = Vector3dVector(current_patch)
                    if (show_part_inter):
                        part_no=iou_oids[part_inter_no]                      
                        if(j in part_viz):
                            pcd_list_inter[i][j].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
                        else:
                            pcd_list_inter[i][j].paint_uniform_color([0.8,0.8,0.8])
                    else:    
                        pcd_list_inter[i][j].paint_uniform_color([colors[j,0], colors[j,1], colors[j,2]])    
                    if(viz_p): 
                        vis.add_geometry(pcd_list_inter[i][j])
    
    
    
            save_p_point=PointCloud()
            for j in range (viz_caps_no):
                pcd_list_source[j].transform(flip_transforms)
                pcd_list_target[j].transform(flip_transformt)
                if(save_p):
                    save_p_point+=pcd_list_source[j]
                    save_p_point+=pcd_list_target[j]
                    
            for r in range(inter_models_number):
                flip_transform_inter  = [[1, 0, 0,transform_range[r]],[0,cosval, -sinval,-2],[0,sinval, cosval, 0],[0, 0, 0, 1]]
#                flip_transform_inter = [[cosval, 0, sinval,-transform_range[r]],[0, 1, 0, -2],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
                for k in range (viz_caps_no):
                    pcd_list_inter[r][k].transform(flip_transform_inter)
                    if(save_p):
                        save_p_point+=pcd_list_inter[r][k]
            if(viz_p):
                vis.update_geometry()
                vis.poll_events()
                vis.update_renderer()
            if(save_p):
                draw_geometries([save_p_point])
#                
                
                soure_name_split=fn[0].split('/')
                soure_name=soure_name_split[-1]
                target_name_split=fn[1].split('/')
                target_name=target_name_split[-1]
#                ply_file_folder='/media/zhao/SSD250/figure_cvpr/interpolation/'+opt.class_choice
                ply_file_folder='/home/zhao/Code/supp/inter_an/Airplane/'
                if not os.path.exists(ply_file_folder):
                    os.makedirs(ply_file_folder);                        
                ply_filename=ply_file_folder+'/'+soure_name+'_'+target_name+'.ply'     
#                draw_geometries([save_p_point])
#                write_point_cloud(ply_filename, save_p_point)
                file_no+=1
                print(soure_name+'_'+target_name)
            
    #        for j in range (viz_caps_no):
    #            pcd_list_source[j].transform(flip_transforms_r)
    #            pcd_list_target[j].transform(flip_transformt_r)
    #        for r in range(inter_models_number):
    #            flip_transform_inter_r  = [[1, 0, 0, -transform_range[r]],[0, 1, 0,2],[0, 0, 1,0],[0, 0, 0, 1]]
    ##                    flip_transform_inter_r   = [[1, 0, 0,transform_range[r]],[0, 1, 0, 2],[0, 0, 1,0],[0, 0, 0, 1]]
    #            for k in range (viz_caps_no):
    #                pcd_list_inter[r][k].transform(flip_transform_inter_r)
            if(viz_p):    
                time.sleep(1)
                                    
    
    if(viz_p):  
        vis.destroy_window()

