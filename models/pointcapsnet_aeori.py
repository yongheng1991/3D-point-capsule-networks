#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:08:48 2018

@author: zhao
"""


import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
#import pdb
import torch.nn.functional as F
from collections import OrderedDict
#from datasets import PartDataset
#from open3d import *
#from tensorboardX import SummaryWriter
from torch.autograd.gradcheck import gradgradcheck, gradcheck
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, 'nndistance'))
from modules.nnd import NNDModule
distChamfer = NNDModule()

#from emdmodules.emd import EMDModule
#distEMD = EMDModule()

#sys.path.append(BASE_DIR)
#from nndistance.modules.nnd import NNDModule
#distChamfer = NNDModule()

#sys.path.append(os.path.join(BASE_DIR, 'emdistance'))
#from emdistance.modules.emd import EMDModule
#distEMD = EMDModule()


USE_CUDA = True


class ConvLayer(nn.Module):
    def __init__(self, num_points=2048,input_channel=3):
        super(ConvLayer, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)
        # self.mp1 = torch.nn.MaxPool1d(num_points)
        # self.num_points = num_points
        # self.global_feat = global_feat

    def forward(self, x):
        # batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
        # x = self.bn3(self.conv3(x))
        # x = self.mp1(x)
        # x = x.view(-1, 1024)
        # if self.global_feat:
        #     return x, trans
        # else:
        #     x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        #     return torch.cat([x, pointfeat], 1), trans


class PrimaryCaps(nn.Module):
    def __init__(self, prim_vec_size=8, num_points=2048):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, 1024, 1)),
                ('bn3', nn.BatchNorm1d(1024)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(prim_vec_size)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=2)
        return self.squash(u.squeeze())
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
#It is wrong to use 2d conv, it is the wront comparison    
class ConvsEncoder(nn.Module):
    def __init__(self, prim_vec_size=16, num_points=2048):
        super(ConvsEncoder, self).__init__()

        self.prim_convs = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, 1024, 1)),
                ('bn3', nn.BatchNorm1d(1024)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(prim_vec_size)])
        self.digit_convs= nn.Conv2d(16, 64, (1024-64+1,1))
        self.bn= nn.BatchNorm2d(64)
        
    def forward(self, x):
        u = [capsule(x) for capsule in self.prim_convs]
        u = torch.stack(u, dim=2)
        u=F.relu(self.bn(self.digit_convs(u.transpose(1,2))))
        return u.transpose(1,2).squeeze()

    
class PointnetLikeEncoder(nn.Module):
    def __init__(self,prim_cap_size=1024, prim_vec_size=16,digit_vec_size=64, digit_cap_size=48, num_points=2048):
        super(PointnetLikeEncoder, self).__init__()

        self.prim_convs = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, prim_cap_size, 1)),
                ('bn3', nn.BatchNorm1d(prim_cap_size)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(prim_vec_size)])
    
        self.digit_convs = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(prim_vec_size, digit_cap_size, 1)),
                ('bn3', nn.BatchNorm1d(digit_cap_size)),
                ('mp1', torch.nn.MaxPool1d(prim_cap_size)),
            ]))
            for _ in range(digit_vec_size)])        
        
        
    def forward(self, x):
        u = [prim_capsule(x) for prim_capsule in self.prim_convs]
        u = torch.stack(u, dim=2).squeeze()
        u = [digit_capsule(u.transpose(1,2)) for digit_capsule in self.digit_convs]
        u = torch.stack(u, dim=2).squeeze()               
#        u=F.relu(self.bn(self.digit_convs(u.transpose(1,2))))
        return u
    

class DigitCaps(nn.Module):
    def __init__(self, digit_caps_size=16, prim_caps_size=1024, prim_vec_size=16, digit_vec_size=64):
        super(DigitCaps, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.digit_caps_size = digit_caps_size
        self.W = nn.Parameter(torch.randn(digit_caps_size, prim_caps_size, digit_vec_size, prim_vec_size))

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = Variable(torch.zeros(x.size(0), self.digit_caps_size, self.prim_caps_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor






class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2048):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

    
class AtlasCapsDecoder(nn.Module):
    def __init__(self, digit_caps_size, digit_vec_size, num_points):
        super(AtlasCapsDecoder, self).__init__()
        self.digit_caps_size = digit_caps_size
        self.bottleneck_size=digit_vec_size
        self.num_points = num_points
        self.nb_primitives=int(num_points/digit_caps_size)
#        self.nb_primitives=64
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size+2) for i in range(0, self.nb_primitives)])
    def forward(self, x, data):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(
                x.size(0), 2, self.digit_caps_size))
            rand_grid.data.uniform_(0, 1)
#            if(i<3):
#                print(rand_grid[0,0,0:10])
            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))

        return torch.cat(outs, 2).contiguous()

    def forward_inference(self, x, grid):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()



class AtlasCapsDecoderUniformGrid(nn.Module):
    def __init__(self, digit_caps_size, digit_vec_size, num_points):
        super(AtlasCapsDecoderUniformGrid, self).__init__()
        self.digit_caps_size = digit_caps_size
        self.bottleneck_size=digit_vec_size
        self.num_points = num_points
        self.nb_primitives=int(num_points/digit_caps_size)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size+2) for i in range(0, self.nb_primitives)])
        grid_size=[8,4]
        sub_square_grid_width=8
        batch_size=8
        self.grid=torch.FloatTensor(int(grid_size[0]*grid_size[1]),batch_size,2,int(sub_square_grid_width*sub_square_grid_width))

#        self.grid=torch.zero(int(grid_size[0]*grid_size[1]),batch_size,int(sub_square_grid_width*sub_square_grid_width),int(sub_square_grid_width*sub_square_grid_width))
        for m in range (grid_size[0]):
            for n in range (grid_size[1]):
                tmp_x=torch.range(m*sub_square_grid_width, (m+1)*sub_square_grid_width-1)
                tmp_y=torch.range(n*sub_square_grid_width, (n+1)*sub_square_grid_width-1)
                grid_point_x=(tmp_x.view(1,sub_square_grid_width).repeat(sub_square_grid_width,1).view(1,1,sub_square_grid_width*sub_square_grid_width).repeat(batch_size,1,1))/64
                grid_point_y=(tmp_y.view(sub_square_grid_width,1).repeat(1,8).view(1,1,sub_square_grid_width*sub_square_grid_width).repeat(batch_size,1,1))/64
                grid_point=torch.cat((grid_point_x,grid_point_y),1)
                self.grid[0,]=grid_point

                
    def forward(self, x, data):
        outs = []
        for i in range(0, self.nb_primitives):
#            rand_grid = Variable(torch.cuda.FloatTensor(
#                x.size(0), 2, self.digit_caps_size))
#            rand_grid.data.uniform_(0, 1)
            current_grid=Variable(self.grid[0].cuda())
            y = torch.cat((current_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()


class AtlasCapsDecoderNoGrid(nn.Module):
    def __init__(self, digit_caps_size, digit_vec_size, num_points):
        super(AtlasCapsDecoderNoGrid, self).__init__()
        self.digit_caps_size = digit_caps_size
        self.bottleneck_size=digit_vec_size
        self.num_points = num_points
        self.nb_primitives=int(num_points/digit_caps_size)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size) for i in range(0, self.nb_primitives)])
    def forward(self, x, data):
        outs = []
        for i in range(0, self.nb_primitives):
#            rand_grid = Variable(torch.cuda.FloatTensor(
#                x.size(0), 2, self.digit_caps_size))
#            rand_grid.data.uniform_(0, 1)
#            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](x.transpose(2, 1)))
        return torch.cat(outs, 2).contiguous()

#    def forward_inference(self, x, grid):
#        outs = []
#        for i in range(0, self.nb_primitives):
#            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
#            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
#            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
#            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
#            outs.append(self.decoder[i](y))
#        return torch.cat(outs, 2).contiguous()





class AtlasCapsDecoder2(nn.Module):
    def __init__(self, digit_caps_size,digit_vec_size, num_points):
        super(AtlasCapsDecoder2, self).__init__()
        self.digit_caps_size = digit_caps_size
        self.digit_vec_size = digit_vec_size
        self.bottleneck_size=digit_vec_size
        self.num_points = num_points
        self.nb_primitives=digit_caps_size
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size+2) for i in range(0, self.nb_primitives)])
    def forward(self, x, data):
                
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(
                x.size(0), 2, (self.num_points/self.nb_primitives)))
            rand_grid.data.uniform_(0, 1)
            y = x[:,i,:].unsqueeze(2).expand(x.size(0), x.size(2), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()

class AtlasCapsSegDecoder(nn.Module):
    def __init__(self, digit_caps_size,digit_vec_size, num_points,parts):
        super(AtlasCapsSegDecoder, self).__init__()
        self.digit_caps_size = digit_caps_size
        self.digit_vec_size = digit_vec_size
        self.bottleneck_size=digit_vec_size
        self.num_points = num_points
        self.nb_primitives=int(num_points/(digit_caps_size/parts))
        self.parts=parts
        self.decoder = nn.ModuleList([nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size+2) for i in range(0, self.nb_primitives)]) for p in range(0, parts)]) 
    def forward(self, x,data):
        outs = []
        for p in range(self.parts):
            x_p=x[:, p*16:(p+1)*16, :]
            for i in range(self.nb_primitives):
                rand_grid = Variable(torch.cuda.FloatTensor(
                    x_p.size(0), 2, int(self.digit_caps_size/self.parts)))
                rand_grid.data.uniform_(0, 1)
                y = torch.cat((rand_grid, x_p.transpose(2, 1)), 1).contiguous()    
                outs.append(self.decoder[p][i](y))
        return torch.cat(outs, 2).contiguous()
    


    
class PointCapsNet(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',input_channel=3):
        super(PointCapsNet, self).__init__()
        self.conv_layer = ConvLayer(num_points=num_points, input_channel=input_channel)
        self.primary_capsules = PrimaryCaps(prim_vec_size, num_points)
        self.digit_capsules = DigitCaps(digit_caps_size, prim_caps_size, prim_vec_size, digit_vec_size)
        if (ae=='ae1'):
            self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)
        elif (ae=='ae2'):
            self.decoder = AtlasCapsDecoder2(digit_caps_size,digit_vec_size, num_points)
        elif(ae=='uniform_grid'):
            self.decoder = AtlasCapsDecoderUniformGrid(digit_caps_size,digit_vec_size, num_points)
        elif (ae=='no_grid'):
            self.decoder = AtlasCapsDecoderNoGrid(digit_caps_size,digit_vec_size, num_points)
    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_capsules(x1)
        output = self.digit_capsules(x2)
        reconstructions = self.decoder(output, data)
        return output, reconstructions
    
    def forward_inference(self, data, grid):
        x1 = self.conv_layer(data)
        x2 = self.primary_capsules(x1)
        output = self.digit_capsules(x2)        
        reconstructions = self.decoder.forward_inference(output, grid)
        return output, reconstructions
    
    def forward_instance(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_capsules(x1)
        output = self.digit_capsules(x2)
#        reconstructions = self.decoder(output, data)
        return output
    
    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss 

class PointCapsNetDecoder(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae2'):
        super(PointCapsNetDecoder, self).__init__()
#        self.conv_layer = ConvLayer()
#        self.primary_capsules = PrimaryCaps(prim_vec_size, num_points)
#        self.digit_capsules = DigitCaps(digit_caps_size, prim_caps_size, prim_vec_size, digit_vec_size)
        if (ae=='ae1'):
            self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)
        elif (ae=='ae2'):
            self.decoder = AtlasCapsDecoder2(digit_caps_size,digit_vec_size, num_points)
    def forward(self, data, output):
        reconstructions = self.decoder(data, output)
        return  reconstructions

#It is wrong to use 2d conv in 'ConvsEncoder', it is the wront comparison    
class PointConvsNet(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',input_channel=3):
        super(PointConvsNet, self).__init__()
        self.conv_layer = ConvLayer(num_points=num_points, input_channel=input_channel)
        self.encoder= ConvsEncoder(num_points=num_points)
        self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)

    def forward(self, data):
        x1 = self.conv_layer(data)
        output = self.encoder(x1)
        reconstructions = self.decoder(output, data)
        return output, reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss 

#replace the 2d conv in the wrong comparison. Use the point net like to generate 64*64.
class PointConvsNetNew(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',input_channel=3):
        super(PointConvsNetNew, self).__init__()
        self.conv_layer = ConvLayer(num_points=num_points, input_channel=input_channel)
        self.encoder= PointnetLikeEncoder(num_points=num_points,prim_cap_size=1024, prim_vec_size=16,digit_vec_size=64, digit_cap_size=64)        
        self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)

    def forward(self, data):
        x1 = self.conv_layer(data)
        output = self.encoder(x1)
        reconstructions = self.decoder(output, data)
        return output, reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss 

#do it as the reviewer suggest. 3-16-64, do 64 maxpooling to get 64*64 capsules and do the decoding.
class PointConvsNetNew2(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',input_channel=3):
        super(PointConvsNetNew2, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(16, 64, 1)),
                ('bn3', nn.BatchNorm1d(64)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(64)])
    
        self.conv_layer = ConvLayer(num_points=num_points, input_channel=input_channel)
#        self.encoder= PointnetLikeEncoder(num_points=num_points,prim_cap_size=1024, prim_vec_size=16,digit_vec_size=64, digit_cap_size=64)        
        self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
    def forward(self, data):
        data = F.relu(self.bn1(self.conv1(data)))
        u = [capsule(data) for capsule in self.capsules]
        u = torch.stack(u, dim=2)    
#        output=self.squash(u.squeeze())
#        x1 = self.conv_layer(data)
#        output = self.encoder(x1)
        reconstructions = self.decoder(u.squeeze(), data)
        return u.squeeze(), reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss 
    

#do maxpooling 64 times, decrease the dimention of feature from 1024 to 64
class PointConvsNetNew3(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',input_channel=3):
        super(PointConvsNetNew3, self).__init__()
        
        self.conv_layer = ConvLayer(num_points=num_points, input_channel=input_channel)
        self.primary_capsules = PrimaryCaps(prim_vec_size=64, num_points=2048)         
        self.conv_layer2 = torch.nn.Conv2d(1, 1, (1024-64+1,1))
#        self.encoder= PointnetLikeEncoder(num_points=num_points,prim_cap_size=1024, prim_vec_size=16,digit_vec_size=64, digit_cap_size=64)        
        self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_capsules(x1)
        u =self.conv_layer2(x2.unsqueeze(1))    
#        output=self.squash(u.squeeze())

        reconstructions = self.decoder(u.squeeze(), data)
        return u.squeeze(), reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss 
    
#do maxpooling 64 times, decrease the dimention of feature from 1024 to 64, use pointnet liked FC projection
class PointConvsNetNew4(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',input_channel=3):
        super(PointConvsNetNew4, self).__init__()        
        self.conv_layer = ConvLayer(num_points=num_points, input_channel=input_channel)
        self.primary_capsules = PrimaryCaps(prim_vec_size=64, num_points=2048)         
        self.conv_layer2 = torch.nn.Conv1d(1024, 64, 1)
        self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_capsules(x1)
        output =self.conv_layer2(x2).transpose(1,2)
        output=self.squash(output)# it doesn't effect the distribution
        reconstructions = self.decoder(output, data)
        return output, reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss 
    
    

class PointConvsNetDecoder(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',input_channel=3):
        super(PointConvsNetDecoder, self).__init__()
        self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)

    def forward(self, data,output):
        reconstructions = self.decoder(data, output)
        return reconstructions


    
class PointCapsNetSeg(nn.Module):    
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',parts=4):
        super(PointCapsNetSeg, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps(prim_vec_size, num_points*parts)
        self.digit_capsules = DigitCaps(digit_caps_size, prim_caps_size, prim_vec_size, digit_vec_size)
        if (ae=='ae1'):
            self.decoder=AtlasCapsSegDecoder(digit_caps_size,digit_vec_size, num_points, parts)
        elif (ae=='ae2'):
            self.decoder = AtlasCapsDecoder2(digit_caps_size,digit_vec_size, num_points*parts)
        self.num_points=num_points
    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_capsules(x1)
        output = self.digit_capsules(x2)
        reconstructions = self.decoder(output, data)        
        return output, reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist_w1, dist_w2 = distChamfer(data_[:,0:self.num_points,:], reconstructions_[:,0:self.num_points,:])
        dist_b1, dist_b2 = distChamfer(data_[:,self.num_points:self.num_points*2,:], reconstructions_[:,self.num_points:self.num_points*2,:])
        dist_t1, dist_t2 = distChamfer(data_[:,self.num_points*2:self.num_points*3,:], reconstructions_[:,self.num_points*2:self.num_points*3,:])
        dist_e1, dist_e2 = distChamfer(data_[:,self.num_points*3:self.num_points*4,:], reconstructions_[:,self.num_points*3:self.num_points*4,:])
        loss = torch.mean(dist_w1) +torch.mean(dist_w2)+torch.mean(dist_b1) +torch.mean(dist_b2)+torch.mean(dist_t1)+torch.mean(dist_t2)+torch.mean(dist_e1)+torch.mean(dist_e2)  
        return loss
    




class PointCapsNetSeg2Loss(nn.Module):    
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',parts=4):
        super(PointCapsNetSeg2Loss, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps(prim_vec_size, num_points*parts)
        self.digit_capsules = DigitCaps(digit_caps_size, prim_caps_size, prim_vec_size, digit_vec_size)
        self.part_capsules = DigitCaps(parts, digit_caps_size, digit_vec_size, digit_vec_size*2)
        
        if (ae=='ae1'):
            self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)
        elif (ae=='ae2'):
            self.decoder = AtlasCapsDecoder2(digit_caps_size,digit_vec_size, num_points*parts)
        
        self.partdecoder=AtlasCapsDecoder2(parts, digit_vec_size*2, num_points*parts)
            
        self.num_points=num_points
    def forward(self, data):
        x1 = self.conv_layer(data).detach()
        x2 = self.primary_capsules(x1).detach()
        output = self.digit_capsules(x2).detach()
        reconstructions = self.decoder(output, data).detach()  
        
        output_part=self.part_capsules(output)
        reconstructions_part = self.partdecoder(output_part, data)   
        
        return output, reconstructions,output_part,reconstructions_part

    def loss(self, data, reconstructions,reconstructions_part):
         return self.reconstruction_loss(data, reconstructions,reconstructions_part)

    def reconstruction_loss(self, data, reconstructions,reconstructions_part):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        reconstructions_part_= reconstructions_part.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        
        dist_w1, dist_w2 = distChamfer(data_[:,0:self.num_points,:], reconstructions_part_[:,0:self.num_points,:])
        dist_b1, dist_b2 = distChamfer(data_[:,self.num_points:self.num_points*2,:], reconstructions_part_[:,self.num_points:self.num_points*2,:])
        dist_t1, dist_t2 = distChamfer(data_[:,self.num_points*2:self.num_points*3,:], reconstructions_part_[:,self.num_points*2:self.num_points*3,:])
        dist_e1, dist_e2 = distChamfer(data_[:,self.num_points*3:self.num_points*4,:], reconstructions_part_[:,self.num_points*3:self.num_points*4,:])
        
        loss = torch.mean(dist1) +torch.mean(dist2)+torch.mean(dist_w1) +torch.mean(dist_w2)+torch.mean(dist_b1) +torch.mean(dist_b2)+torch.mean(dist_t1)+torch.mean(dist_t2)+torch.mean(dist_e1)+torch.mean(dist_e2)  
#        loss = torch.mean(dist1) +torch.mean(dist2)
        return loss/5



class PointCapsNetPartLabel(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points, ae='ae1',input_channel=3):
        super(PointCapsNetPartLabel, self).__init__()
        self.conv_layer = ConvLayer(num_points=num_points, input_channel=input_channel)
        self.primary_capsules = PrimaryCaps(prim_vec_size, num_points)
        self.digit_capsules = DigitCaps(digit_caps_size, prim_caps_size, prim_vec_size, digit_vec_size)
        self.num_points=num_points
        if (ae=='ae1'):
            self.decoder = AtlasCapsDecoder(digit_caps_size,digit_vec_size, num_points)
            self.part_re = Variable(torch.cuda.FloatTensor(8, self.num_points,1))
            for i in range(32):
                self.part_re[:,(i*64):i*64+16,:]=torch.tensor([[1]])
                self.part_re[:,(i*64)+16:i*64+32,:]=torch.tensor([[2]])
                self.part_re[:,(i*64)+32:i*64+48,:]=torch.tensor([[3]])
                self.part_re[:,(i*48)+48:i*64+64,:]=torch.tensor([[4]])
            
            self.part_label= Variable(torch.cuda.FloatTensor(8, self.num_points,1))
            self.part_label[:,0:512,:]=torch.tensor([[1]])
            self.part_label[:,512:1024,:]=torch.tensor([[2]])
            self.part_label[:,1024:1536,:]=torch.tensor([[3]])
            self.part_label[:,1536:2048,:]=torch.tensor([[4]])
        elif (ae=='ae2'):
            self.decoder = AtlasCapsDecoder2(digit_caps_size,digit_vec_size, num_points)
            self.part_re = Variable(torch.cuda.FloatTensor(8, self.num_points,1))
            self.part_re[:,0:512,:]=torch.tensor([[1]])
            self.part_re[:,512:1024,:]=torch.tensor([[2]])
            self.part_re[:,1024:1536,:]=torch.tensor([[3]])
            self.part_re[:,1536:2048,:]=torch.tensor([[4]])
            
            self.part_label= Variable(torch.cuda.FloatTensor(8, self.num_points,1))
            self.part_label[:,0:512,:]=torch.tensor([[1]])
            self.part_label[:,512:1024,:]=torch.tensor([[2]])
            self.part_label[:,1024:1536,:]=torch.tensor([[3]])
            self.part_label[:,1536:2048,:]=torch.tensor([[4]])

        
    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_capsules(x1)
        output = self.digit_capsules(x2)
        reconstructions = self.decoder(output, data)
        return output, reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data,reconstructions)

    def reconstruction_loss(self, data,reconstructions):

        data_ = data.transpose(2, 1).contiguous()
        data_=torch.cat((data_, self.part_label),2)
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        reconstructions_=torch.cat((reconstructions_,self.part_re),2)
        
        dist1, dist2= distChamfer(data_, reconstructions_)
        loss = (torch.max(dist1)) + (torch.max(dist2))
        return loss 



if __name__ == '__main__':
    USE_CUDA = True
    batch_size=2
    prim_caps_size=1024
    prim_vec_size=16
    digit_caps_size=64
    digit_vec_size=32
    num_points=2048
    
#    point_caps_ae = PointConvsNet(prim_caps_size,prim_vec_size,digit_caps_size,digit_vec_size,num_points)
#    point_caps_ae=point_caps_ae.cuda()
#    rand_data=torch.rand(batch_size, 2048, 3) 
#    rand_data = Variable(rand_data)
#    rand_data = rand_data.transpose(2, 1)
#    rand_data=rand_data.cuda()
#    
#    codewords,reconstruction=point_caps_ae(rand_data)
    
    
    
#    p1 = torch.rand(10,100,4)
#    p2 = torch.rand(10,100,4)
#    points1 = Variable(p1,requires_grad = True)
#    points2 = Variable(p2)
#    points1=points1.cuda()
#    points2=points2.cuda()
#    dist1, dist2 = distChamfer(points1, points2)
#    loss = (torch.mean(dist1)) + (torch.mean(dist2))
#
#    print(loss )
#    
#    prim_caps_size=1024
#    prim_vec_size=16
#    
#    digit_caps_size=64
#    digit_vec_size=64
#    
#    num_points=256
#    part=4
    
    
    
#    point_caps_ae = PointCapsNet(prim_caps_size,prim_vec_size,digit_caps_size,digit_vec_size,num_points)
#    point_caps_ae=torch.nn.DataParallel(point_caps_ae).cuda()
#    
#    rand_data=torch.rand(batch_size, num_points, 3) 
#    rand_data = Variable(rand_data)
#    rand_data = rand_data.transpose(2, 1)
#    rand_data=rand_data.cuda()
#    
#    codewords,reconstruction=point_caps_ae(rand_data)
#   
#    rand_data_ = rand_data.transpose(2, 1).contiguous()
#    reconstruction_ = reconstruction.transpose(2, 1).contiguous()
#
#    dist1, dist2 = distChamfer(rand_data_, reconstruction_)
#    loss = (torch.mean(dist1)) + (torch.mean(dist2))
#    print(loss.item())
    
    

#    point_caps_ae = PointCapsNetSeg(prim_caps_size,prim_vec_size,digit_caps_size,digit_vec_size, num_points)
#    point_caps_ae=torch.nn.DataParallel(point_caps_ae).cuda()
#    
#    rand_data=torch.rand(batch_size, num_points*part, 3)
#    rand_data = Variable(rand_data)
#    rand_data = rand_data.transpose(2, 1)
#    rand_data=rand_data.cuda()
   #    gradgradcheck(point_caps_ae, [rand_data])

#    with SummaryWriter(comment='point_caps_ae') as w:
#        w.add_graph(point_caps_ae, (rand_data,))
        
#    codewords,reconstruction=point_caps_ae(rand_data)