import os
import sys
import numpy as np
import h5py
#import provider

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(BASE_DIR)
#ROOT_DIR = BASE_DIR
#sys.path.append(os.path.join(ROOT_DIR, 'utils'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/'))


class Saved_latent_caps_loader(object):
    def __init__(self, dataset, batch_size=32, npoints=1024, with_seg=False, shuffle=True, train=False,resized=False):
        if(with_seg):
            if train:
                self.h5_file=os.path.join(dataset_main_path,dataset,'latent_caps',"saved_train_with_part_label.h5")
            else:
                self.h5_file=os.path.join(dataset_main_path,dataset,'latent_caps',"saved_test_with_part_label.h5")
        else:
            if train:
                self.h5_file=os.path.join(dataset_main_path,dataset,'latent_caps',"saved_train_wo_part_label.h5")
            else:
                self.h5_file=os.path.join(dataset_main_path,dataset,'latent_caps',"saved_test_wo_part_label.h5")

        if(resized):
            self.h5_file=self.h5_file+'_resized.h5'
             
            
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.with_seg = with_seg
#        self.h5_files = getDataFiles(self.list_filename)
#        self.h5_file = getDataFiles(self.list_filename)
        self.reset()
    def reset(self):
#        ''' reset order of h5 files '''
#        self.file_idxs = np.arange(0, len(self.h5_files))
#        if self.shuffle:
#            np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_cls_label = None
        self.current_part_label = None
#        self.current_file_idx = 0
        self.batch_idx = 0


#    def _get_data_filename(self):
#        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _load_data_file(self, filename):
        if self.with_seg:
            self.current_data, self.current_part_label ,self.current_cls_label= self.load_h5(filename)
            self.current_cls_label = np.squeeze(self.current_cls_label)
            self.batch_idx = 0                 
            if self.shuffle:
                self.current_data, self.current_part_label, self.current_cls_label,_ = self.shuffle_data(
                    self.current_data, self.current_part_label, self.current_cls_label)
        else:           
            self.current_data, self.current_cls_label = self.load_h5(filename)
            self.current_cls_label = np.squeeze(self.current_cls_label)
            self.batch_idx = 0                  
            if self.shuffle:
                self.current_data, self.current_cls_label, _ = self.shuffle_data(
                    data=self.current_data,part_labels=[], cls_labels=self.current_cls_label)

    def _has_next_batch_in_file(self):
        return self.batch_idx*self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        if (self.current_data is None ):
#            if self.current_file_idx >= len(self.h5_files):
#                return False
            self._load_data_file(self.h5_file)
            self.batch_idx = 0
#            self.current_file_idx += 1
        return self._has_next_batch_in_file()

    def next_batch(self):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.current_data.shape[0])
#        bsize = end_idx - start_idx
#        batch_label = np.zeros((bsize), dtype=np.int32)
        data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
        cls_label_batch = self.current_cls_label[start_idx:end_idx].copy()
        self.batch_idx += 1
        if self.with_seg:
           part_label_batch = self.current_part_label[start_idx:end_idx].copy()
           return data_batch, part_label_batch, cls_label_batch
        else:
           return data_batch, cls_label_batch

    
    def shuffle_data(self, data, part_labels, cls_labels):
        """ Shuffle data and labels.
            Input:
              data: B,N,... numpy array
              label: B,... numpy array
            Return:
              shuffled data, label and shuffle indices
        """
        if self.with_seg:
            idx = np.arange(len(cls_labels))
            np.random.shuffle(idx)
            return data[idx, ...], part_labels[idx, ...],cls_labels[idx], idx
        else:
            idx = np.arange(len(cls_labels))
            np.random.shuffle(idx)
            return data[idx, ...], cls_labels[idx], idx
    
    
    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        cls_label = f['cls_label'][:]
      
        if self.with_seg:
            part_label = f['part_label'][:]
            return (data, part_label,cls_label)
        else:
            return (data, cls_label)



if __name__ == '__main__':
    
    d = Saved_latent_caps_loader('/home/zhao/Code/dataset/pointnet_data/modelnet40_ply_hdf5_2048/')
    print(d.shuffle)
    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
