import torch
import torch.nn as nn
from torch.autograd import Variable

#from modules.nnd import NNDModule
import torch_nndistance as NND

#dist =  NNDModule()

p1 = torch.rand(10,1000,3)
p2 = torch.rand(10,1500,3)
points1 = Variable(p1,requires_grad = True)
points2 = Variable(p2)
points1=points1.cuda()
points2=points2.cuda()
dist1, dist2 = NND.nnd(points1, points2)
print(dist1, dist2)
loss = torch.sum(dist1)
print(loss)
loss.backward()
print(points1.grad, points2.grad)


points1 = Variable(p1.cuda(), requires_grad = True)
points2 = Variable(p2.cuda())
dist1, dist2 = NND.nnd(points1, points2)
print(dist1, dist2)
loss = torch.sum(dist1)
print(loss)
loss.backward()
print(points1.grad, points2.grad)
