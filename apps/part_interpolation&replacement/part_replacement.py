#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:45:51 2018

@author: zhao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:41:58 2018

@author: zhao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 20:10:10 2018

@author: zhao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:35:20 2018

@author: zhao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:59:27 2018

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
parser.add_argument('--part_model', type=str, default='../checkpoints/part_seg_all_codeword_5percent.pth', help='model path')

cat_no={'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 
 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 
 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

parser.add_argument('--class_choice', type=str, default='Lamp', help='aclass_choice ')

part_inter_no=1
viz_p=False
save_p=True

#show_part_inter=True


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


pcd_gt_source=[]
for i in range(2):
    pcd = PointCloud()    
    pcd_gt_source.append(pcd)
    
pcd_gt_target=[]
for i in range(2):
    pcd = PointCloud()    
    pcd_gt_target.append(pcd)    

    
pcd_gt_replace_source=[]
for i in range(2):
    pcd = PointCloud()    
    pcd_gt_replace_source.append(pcd)
    
pcd_gt_replace_target=[]
for i in range(2):
    pcd = PointCloud()    
    pcd_gt_replace_target.append(pcd) 


pcd_caps_replace_source=[]
for i in range(64):
    pcd = PointCloud()    
    pcd_caps_replace_source.append(pcd)
    
pcd_caps_replace_target=[]
for i in range(64):
    pcd = PointCloud()    
    pcd_caps_replace_target.append(pcd) 

    
    

#chair
rotation_angle=-np.pi/2
cosval = np.cos(rotation_angle)
sinval = np.sin(rotation_angle)
flip_transform_gt_s  = [[cosval, 0, sinval,-3],[0, 1, 0,-1],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
flip_transform_gt_t  = [[cosval, 0, sinval,-3],[0, 1, 0,1],[-sinval, 0, cosval,0],[0, 0, 0, 1]]

flip_transform_gt_re_s  = [[cosval, 0, sinval,0],[0, 1, 0,-1],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
flip_transform_gt_re_t  = [[cosval, 0, sinval,0],[0, 1, 0,1],[-sinval, 0, cosval,0],[0, 0, 0, 1]]

flip_transform_caps_re_s  = [[cosval, 0, sinval,3],[0, 1, 0,-1],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
flip_transform_caps_re_t  = [[cosval, 0, sinval,3],[0, 1, 0,1],[-sinval, 0, cosval,0],[0, 0, 0, 1]]



#airplane
#rotation_angle=np.pi/2
#cosval = np.cos(rotation_angle)
#sinval = np.sin(rotation_angle)
#flip_transforms = [[1, 0, 0,-2],[0,cosval, -sinval,1.5],[0,sinval, cosval, 0],[0, 0, 0, 1]]
#flip_transforms_r  = [[1, 0, 0,2],[0, 1, 0,-1.5],[0, 0, 1,0],[0, 0, 0, 1]]
#
#flip_transform_gt_s  = [[1, 0, 0,-3],[0,cosval, -sinval,-1],[0,sinval, cosval, 0],[0, 0, 0, 1]]
#flip_transform_gt_t  = [[1, 0, 0,-3],[0,cosval, -sinval,1],[0,sinval, cosval, 0],[0, 0, 0, 1]]
#
#flip_transform_gt_re_s  = [[1, 0, 0,0],[0,cosval, -sinval,-1],[0,sinval, cosval, 0],[0, 0, 0, 1]]
#flip_transform_gt_re_t  = [[1, 0, 0,0],[0,cosval, -sinval,1],[0,sinval, cosval, 0],[0, 0, 0, 1]]
#
#flip_transform_caps_re_s = [[1, 0, 0,3],[0,cosval, -sinval,-1],[0,sinval, cosval, 0],[0, 0, 0, 1]]
#flip_transform_caps_re_t = [[1, 0, 0,3],[0,cosval, -sinval,1],[0,sinval, cosval, 0],[0, 0, 0, 1]]


if(viz_p):
    vis = Visualizer()
    vis.create_window()
colors = plt.cm.tab20((np.arange(20)).astype(int))

    


n_batch = 0

file_no=0
for epoch in range(4):
    for batch_id, data in enumerate(train_dataloader):
    
        points, part_label, cls_label,fn= data
        if(points.size(0)<opt.batch_size):  
            break
        print ('batch id', batch_id )
        if not (opt.class_choice==None ):
            cls_label[:]= cat_no[opt.class_choice]
     
        all_model_pcd=PointCloud() 
        if (batch_id!=-1): 
    #    if (batch_id==20):  
            
    #        target = cls_label
                    
            gt_source_list0=[]
            gt_source_list1=[]
            gt_target_list0=[]
            gt_target_list1=[]
            for point_id in range(2048):
                if(part_label[0,point_id]==part_inter_no ):
                    gt_source_list0.append(points[0,point_id,:])
                else:
                    gt_source_list1.append(points[0,point_id,:])
                    
                if( part_label[1,point_id]==part_inter_no):
                    gt_target_list0.append(points[1,point_id,:])
                else:
                    gt_target_list1.append(points[1,point_id,:])
    
            # draw GT with colored part
            pcd_gt_source[0].points=Vector3dVector(gt_source_list0)    
            pcd_gt_source[0].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
            pcd_gt_source[0].transform(flip_transform_gt_s)
            all_model_pcd+=pcd_gt_source[0]
            
            
            pcd_gt_source[1].points=Vector3dVector(gt_source_list1)    
            pcd_gt_source[1].paint_uniform_color([0.8,0.8,0.8])
            pcd_gt_source[1].transform(flip_transform_gt_s)
            all_model_pcd+=pcd_gt_source[1]    
            
            
            pcd_gt_target[0].points=Vector3dVector(gt_target_list0)    
            pcd_gt_target[0].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
            pcd_gt_target[0].transform(flip_transform_gt_t)
            all_model_pcd+=pcd_gt_target[0]
            
            
            pcd_gt_target[1].points=Vector3dVector(gt_target_list1)    
            pcd_gt_target[1].paint_uniform_color([0.8,0.8,0.8])
            pcd_gt_target[1].transform(flip_transform_gt_t)
            all_model_pcd+=pcd_gt_target[1]
            
            # replace GT colored parts
            pcd_gt_replace_source[0].points=Vector3dVector(gt_target_list0)    
            pcd_gt_replace_source[0].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
            pcd_gt_replace_source[0].transform(flip_transform_gt_re_s)
            all_model_pcd+=pcd_gt_replace_source[0]
            
            pcd_gt_replace_source[1].points=Vector3dVector(gt_source_list1)    
            pcd_gt_replace_source[1].paint_uniform_color([0.8,0.8,0.8])
            pcd_gt_replace_source[1].transform(flip_transform_gt_re_s)
            all_model_pcd+=pcd_gt_replace_source[1]      
            
            pcd_gt_replace_target[0].points=Vector3dVector(gt_source_list0)    
            pcd_gt_replace_target[0].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
            pcd_gt_replace_target[0].transform(flip_transform_gt_re_t)
            all_model_pcd+=pcd_gt_replace_target[0]
            
            pcd_gt_replace_target[1].points=Vector3dVector(gt_target_list1)    
            pcd_gt_replace_target[1].paint_uniform_color([0.8,0.8,0.8])
            pcd_gt_replace_target[1].transform(flip_transform_gt_re_t)
            all_model_pcd+=pcd_gt_replace_target[1]
            
    #        draw_geometries([pcd_gt_source[0],pcd_gt_source[1],pcd_gt_target[0],pcd_gt_target[1],
    #                         pcd_gt_replace_source[0],pcd_gt_replace_source[1],pcd_gt_replace_target[0],pcd_gt_replace_target[1]])
            if(viz_p): 
                vis.add_geometry(pcd_gt_source[0])       
                vis.add_geometry(pcd_gt_source[1]) 
                vis.add_geometry(pcd_gt_target[0]) 
                vis.add_geometry(pcd_gt_target[1]) 
                vis.add_geometry(pcd_gt_replace_source[0]) 
                vis.add_geometry(pcd_gt_replace_source[1]) 
                vis.add_geometry(pcd_gt_replace_target[0])
                vis.add_geometry(pcd_gt_replace_target[1])
    
    
            #capsule based replacement
            
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
        
    
            capsmap, expand = Variable(capsmap), Variable(expand)
            capsmap,expand = capsmap.cuda(), expand.cuda()
    
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
    #        
    #       saved the index of capsules which are assigned to current part 
            part_no=iou_oids[part_inter_no]
            part_viz=[]
            for caps_no in range (64):
                if(pred_choice[0,caps_no]==part_no and pred_choice[1,caps_no]==part_no):
                    part_viz.append(caps_no)
          
            #replace the capsules
            capsmap_replace=capsmap.clone()
            capsmap_replace= Variable(capsmap_replace)
            capsmap_replace = capsmap_replace.cuda()
            for j in range (len(part_viz)):
                capsmap_replace[0,part_viz[j],]=capsmap[1,part_viz[j],]
                capsmap_replace[1,part_viz[j],]=capsmap[0,part_viz[j],]      
                
            reconstructions_replace = capsule_net_decoder(capsmap_replace, expand)
            reconstructions_replace=reconstructions_replace.transpose(1,2).data.cpu() 
    
            for j in range(64):
                current_patch_s=torch.zeros(32,3)
                current_patch_t=torch.zeros(32,3)
    
                for m in range(32):
                    current_patch_s[m,]=reconstructions_replace[0][64*m+j,]    
                    current_patch_t[m,]=reconstructions_replace[1][64*m+j,]    
                pcd_caps_replace_source[j].points = Vector3dVector(current_patch_s)
                pcd_caps_replace_target[j].points = Vector3dVector(current_patch_t)
                part_no=iou_oids[part_inter_no]                      
                if(j in part_viz):
                    pcd_caps_replace_source[j].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
                    pcd_caps_replace_target[j].paint_uniform_color([colors[6,0], colors[6,1], colors[6,2]])
                else:
                    pcd_caps_replace_source[j].paint_uniform_color([0.8,0.8,0.8])
                    pcd_caps_replace_target[j].paint_uniform_color([0.8,0.8,0.8])
                
                pcd_caps_replace_source[j].transform(flip_transform_caps_re_s)
                pcd_caps_replace_target[j].transform(flip_transform_caps_re_t)
                
                if(viz_p): 
                    vis.add_geometry(pcd_caps_replace_source[j])
                    vis.add_geometry(pcd_caps_replace_target[j])      
                if (save_p):
                    all_model_pcd+=pcd_caps_replace_source[j]
                    all_model_pcd+=pcd_caps_replace_target[j]
    
            if (save_p):
#                draw_geometries([all_model_pcd])      
                ply_file_folder='/media/zhao/SSD250/figure_cvpr/repalcement/'+opt.class_choice
                if not os.path.exists(ply_file_folder):
                    os.makedirs(ply_file_folder);                        
                ply_filename=ply_file_folder+'/part_inter_'+str(file_no)+'.ply'     
                write_point_cloud(ply_filename, all_model_pcd)
                file_no+=1
            if(viz_p):        
                vis.update_geometry()
                vis.poll_events()
                vis.update_renderer()
                time.sleep(1)
    
    if(viz_p):  
        vis.destroy_window()

