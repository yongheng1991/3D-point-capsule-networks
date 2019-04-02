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

from logger import Logger

from pointcapsnet_ae import PointCapsNet

def main():
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_caps_size, opt.num_points)
  
    if opt.model != '':
        capsule_net.load_state_dict(torch.load(opt.model))
 
    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        capsule_net = torch.nn.DataParallel(capsule_net)
        capsule_net.to(device)

    #create folder to save logs
    log_dir='./logs'+'/'+opt.dataset+'_dataset_'+str(opt.latent_caps_size)+'caps_'+str(opt.latent_vec_size)+'vec'+'_batch_size_'+str(opt.batch_size)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir);
    logger = Logger(log_dir)

    #create folder to save trained models
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf);
    
    if opt.dataset=='shapenet_part':
        train_dataset = shapenet_part_loader.PartDataset(classification=True, npoints=opt.num_points, split='train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)        
    elif opt.dataset=='shapenet_core13':
        train_dataset = shapenet_core13_loader.ShapeNet(normal=False, npoints=opt.num_points, train=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    elif opt.dataset=='shapenet_core55':
        train_dataset = shapenet_core55_loader.Shapnet55Dataset(batch_size=opt.batch_size, npoints=opt.num_points, shuffle=True, train=True)


# training process for 'shapenet_part' or 'shapenet_core13'
    capsule_net.train()
    if 'train_dataloader' in locals().keys() :

        for epoch in range(opt.n_epochs):
            if epoch < 50:
                optimizer = optim.Adam(capsule_net.parameters(), lr=0.0001)
            elif epoch<150:
                optimizer = optim.Adam(capsule_net.parameters(), lr=0.00001)
            else:
                optimizer = optim.Adam(capsule_net.parameters(), lr=0.000001)

            train_loss_sum = 0
            for batch_id, data in enumerate(train_dataloader):
                points, _= data
                if(points.size(0)<opt.batch_size):
                    break
                points = Variable(points)
                points = points.transpose(2, 1)
                if USE_CUDA:
                    points = points.cuda()
    
                optimizer.zero_grad()
                latent_caps, reconstructions= capsule_net(points)
                train_loss = capsule_net.module.loss(points, reconstructions)
                train_loss.backward()
                optimizer.step()
                train_loss_sum += train_loss.item()
                
                info = {'train_loss': train_loss.item()}
               
                for tag, value in info.items():
                    logger.scalar_summary(
                        tag, value, (len(train_dataloader) * epoch) + batch_id + 1)                
              
                if batch_id % 50 == 0:
                    print('bactch_no:%d/%d, train_loss: %f ' %  (batch_id, len(train_dataloader), train_loss.item()))
    
            print('Average train loss of epoch %d : %f' %
                  (epoch, (train_loss_sum / len(train_dataloader))))
    
            if epoch% 5 == 0:
                dict_name=opt.outf+'/'+opt.dataset+'_dataset_'+ '_'+str(opt.latent_caps_size)+'caps_'+str(opt.latent_caps_size)+'vec_'+str(epoch)+'.pth'
                torch.save(capsule_net.module.state_dict(), dict_name)


# training process for 'shapenet_core55'
    else:
        for epoch in range(opt.n_epochs):
            if epoch < 20:
                optimizer = optim.Adam(capsule_net.parameters(), lr=0.001)
            elif epoch<50:
                optimizer = optim.Adam(capsule_net.parameters(), lr=0.0001)
            else:
                optimizer = optim.Adam(capsule_net.parameters(), lr=0.00001)
        
            train_loss_sum = 0
            while train_dataset.has_next_batch():    
                batch_id, points_= train_dataset.next_batch()
                points = torch.from_numpy(points_)
                if(points.size(0)<opt.batch_size):
                    break
                points = Variable(points)
                points = points.transpose(2, 1)
                if USE_CUDA:
                    points = points.cuda()
                optimizer.zero_grad()
                latent_caps, reconstructions= capsule_net(points)
                train_loss = capsule_net.module.loss(points, reconstructions)
                train_loss.backward()
                optimizer.step()    
                train_loss_sum += train_loss.item()                
    
                info = {'train_loss': train_loss.item()}
                for tag, value in info.items():
                    logger.scalar_summary(
                        tag, value, (int(57448/opt.batch_size) * epoch) + batch_id + 1)
                nfo = {'train_loss': train_loss.item()}
                for tag, value in info.items():
                    logger.scalar_summary(
                        tag, value, (int(57448/opt.batch_size) * epoch) + batch_id + 1)
                    
                if batch_id % 50 == 0:
                    print('bactch_no:%d/%d at epoch %d train_loss: %f ' %  (batch_id, int(57448/opt.batch_size),epoch,train_loss.item() )) # the dataset size is 57448
            print('Average train loss of epoch %d : %f' % (epoch, (train_loss_sum / int(57448/opt.batch_size))))        
            train_dataset.reset()      
            if epoch% 5 == 0:
                dict_name=opt.outf+'/'+opt.dataset+'_dataset_'+ '_'+str(opt.latent_caps_size)+'caps_'+str(opt.latent_caps_size)+'vec_'+str(epoch)+'.pth'
                torch.save(capsule_net.module.state_dict(), dict_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_core13', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
    opt = parser.parse_args()
    print(opt)
    main()
