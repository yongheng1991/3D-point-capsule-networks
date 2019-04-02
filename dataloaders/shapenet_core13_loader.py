#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 12:32:36 2018

@author: zhao
"""

#from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import sys
from PIL import Image


CHUNK_SIZE = 150
lenght_line = 60
def my_get_n_random_lines(path, n=5):
    MY_CHUNK_SIZE = lenght_line * (n+2)
    lenght = os.stat(path).st_size
    with open(path, 'r') as file:
        file.seek(np.random.randint(400, lenght - MY_CHUNK_SIZE))
        chunk = file.read(MY_CHUNK_SIZE)
        lines = chunk.split(os.linesep)
        return lines[1:n+1]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/shapenet_core13/'))

class ShapeNet(data.Dataset):
    def __init__(self, rootpc=dataset_path, class_choice=None, train=True, npoints=2500, normal=False):
        self.normal = normal
        self.train = train
        self.rootpc = os.path.join(rootpc, 'customShapeNet')
        self.npoints = npoints
        self.datapath = []
        self.catfile = os.path.join(rootpc, 'shapenet_core13_synsetoffset2category.txt')
        self.cat = {}
        self.meta = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        print(self.cat)
        empty = []
        for item in self.cat:
            try:
                dir_point = os.path.join(self.rootpc, self.cat[item], 'ply')
                fns = sorted(os.listdir(dir_point))
            except:
                fns = []
            if train:
                fns = fns[1:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]
                
            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    if(fn[-3:]=='ply'):
                        self.meta[item].append(( os.path.join(dir_point, fn ), item))
            else:
                empty.append(item)
                
        for item in empty:
            del self.cat[item]
        self.idx2cat = {}
        self.size = {}
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            for fn in self.meta[item]:
                self.datapath.append(fn)
                
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        
    def __getitem__(self, index):
        fn = self.datapath[index]
        clss = self.classes[self.datapath[index][1]]
        clss = torch.from_numpy(np.array([clss]).astype(np.int64))
        for i in range(15):
            try:
                mystring = my_get_n_random_lines(fn[0], n=self.npoints)
                point_set = np.loadtxt(mystring).astype(np.float32)
                break
            except ValueError as excep:
                print(fn)
                print(excep)
        if not self.normal:
            point_set = point_set[:, 0:3]
        else:
            point_set[:, 3:6] = 0.1 * point_set[:, 3:6]
 
        return point_set, clss

    def __len__(self):
        return len(self.datapath)
    

if __name__ == '__main__':
    print('Testing Shapenet dataset')
    d = ShapeNet(class_choice=None, train=True, npoints=2500,normal=False,)
    a = len(d)
    tmp=d[0]
    train_dataloader = torch.utils.data.DataLoader(d, batch_size=8, shuffle=True, num_workers=4)
    for data in enumerate(train_dataloader):
        points, _= data

    d = ShapeNet(class_choice=None, train=False, npoints=2500)
    a = a + len(d)
    print(a)
    test=d[0]
