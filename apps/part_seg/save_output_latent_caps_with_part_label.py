#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:51:36 2019

@author: zhao
"""


import argparse
import torch
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../dataloaders')))

import shapenet_part_loader
from open3d import *

from pointcapsnet_ae import PointCapsNet

def main():
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_caps_size, opt.num_points)
  
    if opt.model != '':
        capsule_net.load_state_dict(torch.load(opt.model))
 
    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        capsule_net = torch.nn.DataParallel(capsule_net)
        capsule_net.to(device)
    
    if opt.dataset=='shapenet_part':
        if opt.save_training:
            split='train'
        else :
            split='test'            
        dataset = shapenet_part_loader.PartDataset(classification=False, npoints=opt.num_points, split=split)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)        
        
# init saving process
    pcd = PointCloud() 
    data_size=0
    dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../../dataset'))
    out_file_path=os.path.join(dataset_main_path, opt.dataset,'latent_caps')
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path);   
    if opt.save_training:
        out_file_name=out_file_path+"/saved_train_with_part_label.h5"
    else:
        out_file_name=out_file_path+"/saved_test_with_part_label.h5"        
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    fw = h5py.File(out_file_name, 'w', libver='latest')
    dset = fw.create_dataset("data", (1,opt.latent_caps_size,opt.latent_vec_size,),maxshape=(None,opt.latent_caps_size,opt.latent_vec_size), dtype='<f4')
    dset_s = fw.create_dataset("part_label",(1,opt.latent_caps_size,),maxshape=(None,opt.latent_caps_size,),dtype='uint8')
    dset_c = fw.create_dataset("cls_label",(1,),maxshape=(None,),dtype='uint8')
    fw.swmr_mode = True


#  process for 'shapenet_part' or 'shapenet_core13'
    capsule_net.eval()
    
    for batch_id, data in enumerate(dataloader):
        points, part_label, cls_label= data
        if(points.size(0)<opt.batch_size):
            break
        points = Variable(points)
        points = points.transpose(2, 1)
        if USE_CUDA:
            points = points.cuda()
        latent_caps, reconstructions= capsule_net(points)
       
        # For each resonstructed point, find the nearest point in the input pointset, 
        # use their part label to annotate the resonstructed point,
        # Then after checking which capsule reconstructed this point, use the part label to annotate this capsule
        reconstructions=reconstructions.transpose(1,2).data.cpu()   
        points=points.transpose(1,2).data.cpu()  
        cap_part_count=torch.zeros([opt.batch_size, opt.latent_caps_size, opt.n_classes],dtype=torch.int64)
        for batch_no in range (points.size(0)):
            pcd.points = Vector3dVector(points[batch_no,])
            pcd_tree = KDTreeFlann(pcd)
            for point_id in range (opt.num_points):
                [k, idx, _] = pcd_tree.search_knn_vector_3d(reconstructions[batch_no,point_id,:], 1)
                point_part_label=part_label[batch_no, idx]            
                caps_no=point_id % opt.latent_caps_size
                cap_part_count[batch_no,caps_no,point_part_label]+=1            
        _,cap_part_label=torch.max(cap_part_count,2) # if the reconstucted points have multiple part labels, use the majority as the capsule part label   
 
    
        # write the output latent caps and cls into file
        data_size=data_size+points.size(0)
        new_shape = (data_size,opt.latent_caps_size,opt.latent_vec_size, )
        dset.resize(new_shape)
        dset_s.resize((data_size,opt.latent_caps_size,))
        dset_c.resize((data_size,))
        
        latent_caps_=latent_caps.cpu().detach().numpy()
        target_=cap_part_label.numpy()
        dset[data_size-points.size(0):data_size,:,:] = latent_caps_
        dset_s[data_size-points.size(0):data_size] = target_
        dset_c[data_size-points.size(0):data_size] = cls_label.squeeze().numpy()
    
        dset.flush()
        dset_s.flush()
        dset_c.flush()
        print('accumalate of batch %d, and datasize is %d ' % ((batch_id), (dset.shape[0])))
           
    fw.close()   

    
         

if __name__ == "__main__":
    import h5py
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='../AE/tmp_checkpoints/shapenet_part_dataset__64caps_64vec_70.pth', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='It has to be shapenet part')
#    parser.add_argument('--save_training', type=bool, default=True, help='save the output latent caps of training data or test data')
    parser.add_argument('--save_training', help='save the output latent caps of training data or test data', action='store_true')

    parser.add_argument('--n_classes', type=int, default=16, help='catagories of current dataset')

    opt = parser.parse_args()
    print(opt)
    main()











