from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'nndistance'))
from modules.nnd import NNDModule
distChamfer = NNDModule()
USE_CUDA = True


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class PrimaryPointCapsLayer(nn.Module):
    def __init__(self, prim_vec_size=8, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
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
        if(output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor


class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=16, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
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
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)
        self.th = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class DeConv(nn.Module):
    # latent cap size and vec size must be 64
    def __init__(self):
        super(DeConv, self).__init__()
        # self.deconv1=torch.nn.ConvTranspose1d(64+2,64,2,stride=2)
        # self.deconv2=torch.nn.ConvTranspose1d(64,32,3,stride=1)
        # self.deconv3=torch.nn.ConvTranspose1d(32,16,4,stride=2)
        # self.deconv4=torch.nn.ConvTranspose1d(16,3,5,stride=3)
        # self.deconv5=torch.nn.ConvTranspose1d(4,3,1,stride=1)
        
        self.deconv1=torch.nn.ConvTranspose1d(64+2,64,2,stride=2)
        self.deconv2=torch.nn.ConvTranspose1d(64,32,3,stride=1)
        self.deconv3=torch.nn.ConvTranspose1d(32,16,4,stride=2)
        self.deconv4=torch.nn.ConvTranspose1d(16,4,6,stride=1)
        self.deconv5=torch.nn.ConvTranspose1d(4,3,4,stride=2)
        
        
        
    def forward(self, x):
        out=self.deconv1(x)
        out=self.deconv2(out)
        out=self.deconv3(out)
        out=self.deconv4(out)
        out=self.deconv5(out)
        return out
        
        
class CapsDecoder(nn.Module):
    def __init__(self, latent_caps_size, latent_vec_size, num_points):
        super(CapsDecoder, self).__init__()
        self.latent_caps_size = latent_caps_size
        self.bottleneck_size=latent_vec_size
        self.num_points = num_points
        self.nb_primitives=int(num_points/latent_caps_size)
        self.decoder = nn.ModuleList(
            [DeConv() for i in range(0, self.latent_caps_size)])
        
    def forward(self, x):
        outs = []
        for i in range(0, self.latent_caps_size):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, 1))
            rand_grid.data.uniform_(0, 1)
            y = torch.cat((rand_grid, x[:,i:i+1,:].transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()

    
class PointCapsNet(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, latent_caps_size, latent_vec_size, num_points):
        super(PointCapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(prim_vec_size, num_points)
        self.latent_caps_layer = LatentCapsLayer(latent_caps_size, prim_caps_size, prim_vec_size, latent_vec_size)
        self.caps_decoder = CapsDecoder(latent_caps_size,latent_vec_size, num_points)

    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_point_caps_layer(x1)
        latent_capsules = self.latent_caps_layer(x2)
        reconstructions = self.caps_decoder(latent_capsules)
        return latent_capsules, reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss 
    
# This is a single network which can decode the point cloud from pre-saved latent capsules
class PointCapsNetDecoder(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points):
        super(PointCapsNetDecoder, self).__init__()
        self.caps_decoder = CapsDecoder(digit_caps_size,digit_vec_size, num_points)
    def forward(self, latent_capsules):
        reconstructions = self.caps_decoder(latent_capsules)
        return  reconstructions
    
    
    
def get_cluster_loss(reconstruction, cap_num, points_per_cap):
    dist_loss_sum=0
    for i in range(cap_num):
        points_for_one_cap=reconstruction[:,:,i*points_per_cap:(i+1)*points_per_cap]
        for j in range(points_per_cap):
            point_j=points_for_one_cap[:,:,j:j+1]
            dist_all=torch.sqrt(torch.sum((points_for_one_cap-point_j)**2,dim=1)+1e-8)
            dis_max,_=torch.max(dist_all,dim=-1)
            dist_loss_sum+=torch.mean(dis_max)
    
    dist_loss_sum=dist_loss_sum/(cap_num*points_per_cap)         
    return dist_loss_sum
    
    
    

if __name__ == '__main__':
    USE_CUDA = True
    batch_size=2
    
    prim_caps_size=1024
    prim_vec_size=16
    
    latent_caps_size=64
    latent_vec_size=64
    
    # num_points=2048
    num_points = latent_caps_size*32

    point_caps_ae = PointCapsNet(prim_caps_size,prim_vec_size,latent_caps_size,latent_vec_size,num_points)
    point_caps_ae=torch.nn.DataParallel(point_caps_ae).cuda()
    
    rand_data=torch.rand(batch_size,num_points, 3) 
    rand_data = Variable(rand_data)
    rand_data = rand_data.transpose(2, 1)
    rand_data=rand_data.cuda()
    
    codewords,reconstruction=point_caps_ae(rand_data)
   
    rand_data_ = rand_data.transpose(2, 1).contiguous()
    reconstruction_ = reconstruction.transpose(2, 1).contiguous()

    dist1, dist2 = distChamfer(rand_data_, reconstruction_)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    print(loss.item())
    
    
    cap_num=latent_caps_size
    points_per_cap=int(num_points/cap_num)
    
    cluster_loss = get_cluster_loss(reconstruction, cap_num, points_per_cap)
    print(cluster_loss.item())
    
    
    
    
    
    
    
    
    
    
    
    
    