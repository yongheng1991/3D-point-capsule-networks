#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:35:34 2018

@author: zhao
"""

import argparse
import torch
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../../dataloaders')))
import saved_latent_caps_loader


from pointcapsnet_ae import PointCapsNet
import h5py
from sklearn.svm import LinearSVC



def ResizeDataset(percentage, n_classes, shuffle):
    dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../../dataset'))
    ori_file_name=os.path.join(dataset_main_path, opt.dataset,'latent_caps','saved_train_wo_part_label.h5')           
    out_file_name=ori_file_name+"_resized.h5"
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    fw = h5py.File(out_file_name, 'w', libver='latest')
    dset = fw.create_dataset("data", (1,opt.latent_caps_size,opt.latent_vec_size,),maxshape=(None,opt.latent_caps_size,opt.latent_vec_size), dtype='<f4')
    dset_l = fw.create_dataset("cls_label",(1,),maxshape=(None,),dtype='uint8')
    fw.swmr_mode = True   
    f = h5py.File(ori_file_name)
    data = f['data'][:]
    cls_label = f['cls_label'][:]
    
    #data shuffle
    if shuffle:        
        idx = np.arange(len(cls_label))
        np.random.shuffle(idx)
        data,cls_label = data[idx, ...], cls_label[idx]
    
    class_dist= np.zeros(n_classes)
    for c in range(len(data)):
        class_dist[cls_label[c]]+=1
    print('Ori data to size of :', np.sum(class_dist))
    print ('class distribution of this dataset :',class_dist)
        
    class_dist_new= (percentage*class_dist/100).astype(int)
    for i in range(opt.n_classes):
        if class_dist_new[i]<1 :
            class_dist_new[i]=1
    class_dist_count=np.zeros(n_classes)
    

    data_count=0
    for c in range(len(data)):
        label_c=cls_label[c]
        if(class_dist_count[label_c] < class_dist_new[label_c]):
            class_dist_count[label_c]+=1
            new_shape = (data_count+1,opt.latent_caps_size,opt.latent_vec_size,)
            dset.resize(new_shape)
            dset_l.resize((data_count+1,))
            dset[data_count,:,:] = data[c]
            dset_l[data_count] = cls_label[c]
            dset.flush()
            dset_l.flush()
            data_count+=1
    print('Finished resizing data to size of :', np.sum(class_dist_new))
    print ('class distribution of resized dataset :',class_dist_new)
    fw.close
    
    

def main():

    data_resized=False
    if(opt.percent_training_dataset<100):            
        ResizeDataset( percentage=opt.percent_training_dataset, n_classes=opt.n_classes,shuffle=True)
        data_resized=True
   
    train_dataset =  saved_latent_caps_loader.Saved_latent_caps_loader(
            dataset=opt.dataset, batch_size=opt.batch_size, npoints=opt.num_points, with_seg=False, shuffle=True, train=True,resized=data_resized)
    test_dataset =  saved_latent_caps_loader.Saved_latent_caps_loader(
            dataset=opt.dataset, batch_size=opt.batch_size, npoints=opt.num_points, with_seg=False, shuffle=False, train=False,resized=False)


    train_feature=np.zeros((1,opt.latent_caps_size*opt.latent_vec_size))
    train_label=np.zeros((1,1))
    test_feature=np.zeros((1,opt.latent_caps_size*opt.latent_vec_size))
    test_label=np.zeros((1,1))
    
    batch_id=0    
    while train_dataset.has_next_batch():
        latent_caps, target_ = train_dataset.next_batch()
        train_label=np.concatenate((train_label,target_), axis=None)
        train_label=train_label.astype(int)     
        classes =latent_caps.reshape((latent_caps.shape[0], opt.latent_caps_size*opt.latent_vec_size))        
        train_feature=np.concatenate((train_feature,classes), axis=0)
        batch_id+=1
        if(batch_id % 10 == 0):
            print('add train batch: ', batch_id )
    
    batch_id=0    
    while test_dataset.has_next_batch():
        latent_caps, target_ = test_dataset.next_batch()
        test_label=np.concatenate((test_label,target_), axis=None)
        test_label=test_label.astype(int)     
        classes =latent_caps.reshape((latent_caps.shape[0], opt.latent_caps_size*opt.latent_vec_size))
        test_feature=np.concatenate((test_feature,classes), axis=0)
        batch_id+=1
        if(batch_id % 10 == 0):
            print('add test batch: ', batch_id )
     
    train_feature=train_feature[1:,:]
    train_label=train_label[1:]
    test_feature=test_feature[1:,:]
    test_label=test_label[1:]
      
    print('training the liner SVM.......') 
    clf = LinearSVC()
    clf.fit(train_feature, train_label)
    confidence = clf.score(test_feature, test_label)
    print('Accuracy: %f in percent %d ' % (confidence,opt.percent_training_dataset))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55, modelent40')
    parser.add_argument('--percent_training_dataset', type=int, default=100, help='traing cls with percent of training_data')
    
    parser.add_argument('--n_classes', type=int, default=1, help='catagories of current dataset')

    opt = parser.parse_args()
    if(opt.dataset=='shapenet_part'):
        opt.n_classes=16
    elif (opt.dataset=='shapenet_core13'):
        opt.n_classes=13
    elif (opt.dataset=='shapenet_core55'):
        opt.n_classes=55  
    elif (opt.dataset=='modelent40'):
        opt.n_classes=40
    print(opt)
    
    main()

