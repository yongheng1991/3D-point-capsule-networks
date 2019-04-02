#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 23:15:24 2018

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
import saved_latent_caps_loader


from pointcapsnet_ae import PointCapsNet
from capsule_seg_net import CapsSegNet

import h5py
from sklearn.svm import LinearSVC
import json


def ResizeDataset(percentage, n_classes, shuffle):
    dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../../dataset'))
    ori_file_name=os.path.join(dataset_main_path, opt.dataset,'latent_caps','saved_train_with_part_label.h5')           
    out_file_name=ori_file_name+"_resized.h5"
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    fw = h5py.File(out_file_name, 'w', libver='latest')
    dset = fw.create_dataset("data", (1,opt.latent_caps_size,opt.latent_vec_size,),maxshape=(None,opt.latent_caps_size,opt.latent_vec_size), dtype='<f4')
    dset_s = fw.create_dataset("part_label",(1,opt.latent_caps_size,),maxshape=(None,opt.latent_caps_size,),dtype='uint8')
    dset_c = fw.create_dataset("cls_label",(1,),maxshape=(None,),dtype='uint8')
    fw.swmr_mode = True   
    f = h5py.File(ori_file_name)
    data = f['data'][:]
    part_label = f['part_label'][:]
    cls_label = f['cls_label'][:]
    
    #data shuffle
    if shuffle:        
        idx = np.arange(len(cls_label))
        np.random.shuffle(idx)
        data,part_label,cls_label = data[idx, ...], part_label[idx, ...],cls_label[idx]
    
    class_dist= np.zeros(n_classes)
    for c in range(len(data)):
        class_dist[cls_label[c]]+=1
    print('Ori data to size of :', np.sum(class_dist))
    print ('class distribution of this dataset :',class_dist)
        
    class_dist_new= (percentage*class_dist/100).astype(int)
    for i in range(16):
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
            dset_s.resize((data_count+1,opt.latent_caps_size,))
            dset_c.resize((data_count+1,))
            dset[data_count,:,:] = data[c]
            dset_s[data_count,:] = part_label[c]
            dset_c[data_count] = cls_label[c]
            dset.flush()
            dset_s.flush()
            dset_c.flush()
            data_count+=1
    print('Finished resizing data to size of :', np.sum(class_dist_new))
    print ('class distribution of resized dataset :',class_dist_new)
    fw.close
    
    

def main():
    blue = lambda x:'\033[94m' + x + '\033[0m'
    
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
    objnames = [line.split()[0] for line in lines]
    on2oid = {objcats[i]:i for i in range(len(objcats))}
    fin.close()

    
# load the dataset
    data_resized=False
    if(opt.percent_training_dataset<100):            
        ResizeDataset( percentage=opt.percent_training_dataset, n_classes=16,shuffle=True)
        data_resized=True
   
    train_dataset =  saved_latent_caps_loader.Saved_latent_caps_loader(
            dataset=opt.dataset, batch_size=opt.batch_size, npoints=opt.num_points, with_seg=True, shuffle=True, train=True,resized=data_resized)
    test_dataset =  saved_latent_caps_loader.Saved_latent_caps_loader(
            dataset=opt.dataset, batch_size=opt.batch_size, npoints=opt.num_points, with_seg=True, shuffle=False, train=False,resized=False)


#  load the SemiConvSegNet
    caps_seg_net = CapsSegNet(latent_caps_size=opt.latent_caps_size, latent_vec_size=opt.latent_vec_size , num_classes=opt.n_classes)    
#    if opt.part_model != '':
#        caps_seg_net.load_state_dict(torch.load(opt.part_model))            
    if USE_CUDA:
        caps_seg_net = torch.nn.DataParallel(caps_seg_net).cuda()       
    optimizer = optim.Adam(caps_seg_net.parameters(), lr=0.01)
    
    #create folder to save trained models
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf);
# start training
    n_batch = 0
    for epoch in range(opt.n_epochs):
        batch_id = 0
        caps_seg_net=caps_seg_net.train()
        while train_dataset.has_next_batch():
            latent_caps_, part_label,cls_label = train_dataset.next_batch()            
            
            # translate the part label to one hot
            cur_label_one_hot = np.zeros((len(latent_caps_), 16), dtype=np.float32) # shapnet part has 16 catagories
            for i in range(len(latent_caps_)):
                cur_label_one_hot[i, cls_label[i]] = 1
                iou_oids = object2setofoid[objcats[cls_label[i]]]
                for j in range(opt.latent_caps_size):
                    part_label[i,j]=iou_oids[part_label[i,j]]
            target = torch.from_numpy(part_label.astype(np.int64))

            
            # concatnate the latent caps with the one hot part label
            latent_caps = torch.from_numpy(latent_caps_).float()
            if(latent_caps.size(0)<opt.batch_size):
                continue
            latent_caps, target = Variable(latent_caps), Variable(target)
            if USE_CUDA:
                latent_caps,target = latent_caps.cuda(), target.cuda()                        
            cur_label_one_hot=torch.from_numpy(cur_label_one_hot).float()        
            expand =cur_label_one_hot.unsqueeze(2).expand(len(latent_caps),16,opt.latent_caps_size).transpose(1,2)  
            expand=Variable(expand).cuda()
            latent_caps=torch.cat((latent_caps,expand),2)
    
    
# forward
            optimizer.zero_grad()
            latent_caps=latent_caps.transpose(2, 1)# consider the capsule vector size as the channel in the network
            output_digit=caps_seg_net(latent_caps)
            output_digit = output_digit.view(-1, opt.n_classes)        
     
    
            target= target.view(-1,1)[:,0] 
            train_loss = F.nll_loss(output_digit, target)
            train_loss.backward()
            optimizer.step()
    #        print('bactch_no:%d/%d, train_loss: %f ' % (batch_id, len(train_dataloader)/opt.batch_size, train_loss.item()))
           
            pred_choice = output_digit.data.cpu().max(1)[1]
            correct = pred_choice.eq(target.data.cpu()).cpu().sum()
            
            batch_id+=1
            n_batch=max(batch_id,n_batch)
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, batch_id, n_batch, blue('test'), train_loss.item(), correct.item()/float(opt.batch_size * opt.latent_caps_size)))
         
            
        if epoch % 10 == 0:    
            caps_seg_net=caps_seg_net.eval()    
            correct_sum=0
            batch_id=0
            while test_dataset.has_next_batch():
                latent_caps, part_label,cls_label = test_dataset.next_batch()
                cur_label_one_hot = np.zeros((len(latent_caps), 16), dtype=np.float32)
                for i in range(len(latent_caps)):
                    cur_label_one_hot[i, cls_label[i]] = 1
                    iou_oids = object2setofoid[objcats[cls_label[i]]]
                    for j in range(opt.latent_caps_size):
                        part_label[i,j]=iou_oids[part_label[i,j]]
                target = torch.from_numpy(part_label.astype(np.int64))
        
                latent_caps = torch.from_numpy(latent_caps).float()
                if(latent_caps.size(0)<opt.batch_size):
                    continue
                latent_caps, target = Variable(latent_caps), Variable(target)    
                if USE_CUDA:
                    latent_caps,target = latent_caps.cuda(), target.cuda()
                
                cur_label_one_hot=torch.from_numpy(cur_label_one_hot).float()        
                expand =cur_label_one_hot.unsqueeze(2).expand(len(latent_caps),16,opt.latent_caps_size).transpose(1,2)        
                expand=Variable(expand).cuda()
                latent_caps=torch.cat((latent_caps,expand),2)
                
                
                latent_caps=latent_caps.transpose(2, 1)        
                output=caps_seg_net(latent_caps)
                output = output.view(-1, opt.n_classes)        
                target= target.view(-1,1)[:,0] 
        
#                print('bactch_no:%d/%d, train_loss: %f ' % (batch_id, len(train_dataloader)/opt.batch_size, train_loss.item()))
               
                pred_choice = output.data.cpu().max(1)[1]
                correct = pred_choice.eq(target.data.cpu()).cpu().sum()                
                correct_sum=correct_sum+correct.item()
                batch_id+=1
                
            print(' accuracy of epoch %d is: %f' %(epoch,correct_sum/float((batch_id+1)*opt.batch_size * opt.latent_caps_size)))
            dict_name=opt.outf+'/'+ str(opt.latent_caps_size)+'caps_'+str(opt.latent_caps_size)+'vec_'+ str(opt.percent_training_dataset) + '% of_training_data_at_epoch'+str(epoch)+'.pth'
            torch.save(caps_seg_net.module.state_dict(), dict_name)
             
        train_dataset.reset()
        test_dataset.reset()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='../AE/tmp_checkpoints/shapenet_part_dataset__64caps_64vec_70.pth', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55, modelent40')
    parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
    parser.add_argument('--percent_training_dataset', type=int, default=100, help='traing cls with percent of training_data')
    parser.add_argument('--n_classes', type=int, default=50, help='part classes in all the catagories')

    opt = parser.parse_args()
    print(opt)

    USE_CUDA = True
    main()
    
    





    
   