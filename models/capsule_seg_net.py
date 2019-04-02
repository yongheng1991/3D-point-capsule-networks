
from __future__ import print_function
#import argparse
#import os
#import random
import torch
import torch.nn as nn
#import torch.nn.parallel
import torch.backends.cudnn as cudnn
#import torch.optim as optim
#import torch.utils.data
#import torchvision.transforms as transforms
#import torchvision.utils as vutils
#from torch.autograd import Variable
#from PIL import Image
#import numpy as np
#import matplotlib.pyplot as plt
#import pdb
import torch.nn.functional as F
#from collections import OrderedDict
#import sys

USE_CUDA = True

class CapsSegNet(nn.Module):    
    def __init__(self, latent_caps_size,latent_vec_size , num_classes):
        super(CapsSegNet, self).__init__()
        self.num_classes=num_classes
        self.latent_caps_size=latent_caps_size
        self.seg_convs= nn.Conv1d(latent_vec_size+16, num_classes, 1)    

    def forward(self, data):
        batchsize= data.size(0)
        output = self.seg_convs(data)
        output = output.transpose(2,1).contiguous()
        output = F.log_softmax(output.view(-1,self.num_classes), dim=-1)
        output = output.view(batchsize, self.latent_caps_size, self.num_classes)
        return output
    
 
    
if __name__ == '__main__':
    USE_CUDA = True
   