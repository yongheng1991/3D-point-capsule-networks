#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:52:15 2018

@author: zhao
"""

import os
import sys
import numpy as np
import h5py



def shuffle_data(data):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return data[idx, ...]


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/shapenet_core55/shapenet57448xyzonly.npz'))

class Shapnet55Dataset(object):
    def __init__(self, filename=dataset_path, batch_size=32, npoints=1024, shuffle=True, train=False):
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.total_data=[]
        self._load_data_file(filename)
        self.reset()
    def reset(self):
        ''' reset order of h5 files '''        
        if self.shuffle:
            self.total_data = shuffle_data(self.total_data)
        self.batch_idx = 0


    def _load_data_file(self, filename):
        data_dict = dict(np.load(filename))
        self.total_data = data_dict['data']


    def num_channel(self):
        return 3

    def has_next_batch(self):
        return self.batch_idx*self.batch_size < len(self.total_data)
    
    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m        
        return pc
    
    def next_batch(self):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.total_data))
#        bsize = end_idx - start_idx

        data_batch = self.total_data[start_idx:end_idx, :, :].copy()
        for i in range(self.batch_size):
            data_batch[i,]=self.pc_normalize(data_batch[i,])
            choice = np.random.choice(len(data_batch[i,:,0]), self.npoints, replace=True)
            # resample
            data_batch[i,]= data_batch[i,choice, :]

        self.batch_idx += 1
        return self.batch_idx-1, data_batch


if __name__ == '__main__':
    
    d = Shapnet55Dataset(filename='../../dataset/folding_data/shapenet57448xyzonly.npz',batch_size=8, npoints=2048, shuffle=True,train=False)
    print(d.shuffle)
    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
