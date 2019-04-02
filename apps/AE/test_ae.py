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
import shapenet_part_loader
import shapenet_core13_loader
import shapenet_core55_loader

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
        test_dataset = shapenet_part_loader.PartDataset(classification=True, npoints=opt.num_points, split='test')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)        
    elif opt.dataset=='shapenet_core13':
        test_dataset = shapenet_core13_loader.ShapeNet(normal=False, npoints=opt.num_points, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    elif opt.dataset=='shapenet_core55':
        test_dataset = shapenet_core55_loader.Shapnet55Dataset(batch_size=opt.batch_size,npoints=opt.num_points, shuffle=True, train=False)


# test process for 'shapenet_part' or 'shapenet_core13'
    capsule_net.eval()
    if 'test_dataloader' in locals().keys() :
        test_loss_sum = 0
        for batch_id, data in enumerate(test_dataloader):
            points, _= data
            if(points.size(0)<opt.batch_size):
                break
            points = Variable(points)
            points = points.transpose(2, 1)
            if USE_CUDA:
                points = points.cuda()
            latent_caps, reconstructions= capsule_net(points)
            test_loss = capsule_net.module.loss(points, reconstructions)
            test_loss_sum += test_loss.item()            
            print('accumalate of batch %d loss is : %f' % (batch_id, test_loss.item()))
        test_loss_sum = test_loss_sum / float(len(test_dataloader))
        print('test loss is : %f' % (test_loss_sum))
             
# test process for 'shapenet_core55'
    else:
        test_loss_sum = 0
        while test_dataset.has_next_batch():    
            batch_id, points_= test_dataset.next_batch()
            points = torch.from_numpy(points_)
            if(points.size(0)<opt.batch_size):
                break
            points = Variable(points)
            points = points.transpose(2, 1)
            if USE_CUDA:
                points = points.cuda()
            latent_caps, reconstructions= capsule_net(points)
            test_loss = capsule_net.module.loss(points, reconstructions)
            test_loss_sum += test_loss.item()                
            print('accumalate of batch %d loss is : %f' % (batch_id, test_loss.item()))
        test_loss_sum = test_loss_sum / float(len(test_dataloader))
        print('test loss is : %f' % (test_loss_sum))
                               
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='tmp_checkpoints/shapenet_part_dataset__64caps_64vec_70.pth', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
    opt = parser.parse_args()
    print(opt)
    main()




