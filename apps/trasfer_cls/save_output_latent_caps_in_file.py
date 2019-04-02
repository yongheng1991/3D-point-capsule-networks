#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 23:08:52 2018

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
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../utils')))
import shapenet_part_loader
import shapenet_core13_loader
import shapenet_core55_loader
import modelnet40_loader
from pointcapsnet_ae import PointCapsNet

def main():
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_caps_size, opt.num_points)
  
    if opt.model != '':
        capsule_net.load_state_dict(torch.load(opt.model))
    else:
        print ('pls set the model path')
 
    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        capsule_net = torch.nn.DataParallel(capsule_net)
        capsule_net.to(device)
    
    if opt.dataset=='shapenet_part':
        if opt.save_training:
            split='train'
        else :
            split='test'            
        dataset = shapenet_part_loader.PartDataset(classification=True, npoints=opt.num_points, split=split)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)        
    elif opt.dataset=='shapenet_core13':
        dataset = shapenet_core13_loader.ShapeNet(normal=False, npoints=opt.num_points, train=opt.save_training)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    elif opt.dataset=='shapenet_core55':
        dataset = shapenet_core55_loader.Shapnet55Dataset(batch_size=opt.batch_size,npoints=opt.num_points, shuffle=True, train=opt.save_training)
    elif opt.dataset=='modelnet40':
        dataset = modelnet40_loader.ModelNetH5Dataset(batch_size=opt.batch_size, npoints=opt.num_points, shuffle=True, train=opt.save_training)

        
# init saving process
    data_size=0
    dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../../dataset'))
    out_file_path=os.path.join(dataset_main_path, opt.dataset,'latent_caps')
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path);
   
    if opt.save_training:
        out_file_name=out_file_path+"/saved_train_wo_part_label.h5"
    else:
        out_file_name=out_file_path+"/saved_test_wo_part_label.h5"        
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    fw = h5py.File(out_file_name, 'w', libver='latest')
    dset = fw.create_dataset("data", (1,opt.latent_caps_size,opt.latent_vec_size,),maxshape=(None,opt.latent_caps_size,opt.latent_vec_size), dtype='<f4')
    dset_c = fw.create_dataset("cls_label",(1,),maxshape=(None,),dtype='uint8')
    fw.swmr_mode = True


#  process for 'shapenet_part' or 'shapenet_core13'
    capsule_net.eval()
    if 'dataloader' in locals().keys() :
        test_loss_sum = 0
        for batch_id, data in enumerate(dataloader):
            points, cls_label= data
            if(points.size(0)<opt.batch_size):
                break
            points = Variable(points)
            points = points.transpose(2, 1)
            if USE_CUDA:
                points = points.cuda()
            latent_caps, reconstructions= capsule_net(points)
           
            # write the output latent caps and cls into file
            data_size=data_size+points.size(0)
            new_shape = (data_size,opt.latent_caps_size,opt.latent_vec_size, )
            dset.resize(new_shape)
            dset_c.resize((data_size,))
            
            latent_caps_=latent_caps.cpu().detach().numpy()
            dset[data_size-points.size(0):data_size,:,:] = latent_caps_
            dset_c[data_size-points.size(0):data_size] = cls_label.squeeze().numpy()
        
            dset.flush()
            dset_c.flush()
            print('accumalate of batch %d, and datasize is %d ' % ((batch_id), (dset.shape[0])))
               
        fw.close()   

    
         
#  process for 'shapenet_core55' or 'modelnet40'
    else:
        while dataset.has_next_batch():    
            batch_id, points_= dataset.next_batch()
            points = torch.from_numpy(points_)
            if(points.size(0)<opt.batch_size):
                break
            points = Variable(points)
            points = points.transpose(2, 1)
            if USE_CUDA:
                points = points.cuda()
            latent_caps, reconstructions= capsule_net(points)
            
            data_size=data_size+points.size(0)
            new_shape = (data_size,opt.latent_caps_size,opt.latent_vec_size, )
            dset.resize(new_shape)
            dset_c.resize((data_size,))
            
            latent_caps_=latent_caps.cpu().detach().numpy()
            dset[data_size-points.size(0):data_size,:,:] = latent_caps_
            dset_c[data_size-points.size(0):data_size] = cls_label.squeeze().numpy()
        
            dset.flush()
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
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55, modelnet40')
    parser.add_argument('--save_training', help='save the output latent caps of training data or test data', action='store_true')
    opt = parser.parse_args()
    print(opt)
    main()












