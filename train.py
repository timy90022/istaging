# import tensorflow as tf
import os
from utils.get_data import Image_data
from models import CornerNet
from config import cfg
import numpy as np
import cv2
import torch
from torch.optim import lr_scheduler
import sys
from dataloaders import istg
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

# from test import Debug
class Train():
    def __init__(self):
        # os.environ['CUDA_VISIBLE_DEVICES']='0,1'
        self.net=CornerNet.model()
        self.net.cuda()
        self.net.train()
        
        istg_train = istg.Segmentation(split='train')
        istg_val = istg.Segmentation(split='val')

        self.trainloader = DataLoader(istg_train, batch_size=1, shuffle=True, num_workers=0,drop_last=True)
        self.valloader = DataLoader(istg_train, batch_size=1, shuffle=True, num_workers=0,drop_last=True)

        self.loss    = CornerNet.loss
        self.lr=cfg.learning_rate
        self.weight_decay_rate=cfg.weight_decay_rate
        
        params = list(self.net.parameters()) 

        self.optimizer = torch.optim.Adam([p for p in params if p.requires_grad],
                                        lr=self.lr, betas=(0.5 , 0.999), weight_decay=self.weight_decay_rate)
        
        self.decay_rate=cfg.decay_rate
        self.decay_step=cfg.decay_step

        self.scheduler = lr_scheduler.StepLR(self.optimizer, self.decay_step, gamma=self.decay_rate)


        self.stepsize=cfg.stepsize

        self.snapshot = 10
        self.save_dir = './check/'
        self.all_step = 0
        self.data_quantity = len(self.trainloader)
        self.writer = SummaryWriter()

        self.start_epoch = 0

        if ( os.path.exists(self.save_dir)):
            print("Path exists!")
        else:
            os.mkdir( self.save_dir )

    def train_single(self):
        epoch = 0
        while True:
            epoch_loss = 0
            for step,sample in (enumerate(self.trainloader)):
                self.scheduler.step()
                self.optimizer.zero_grad()
                xs = [x.cuda(non_blocking=True) for x in sample['xs']]
                ys = [x.cuda(non_blocking=True) for x in sample['ys']]
                out = self.net._train(xs)
                loss = self.loss(out,ys)
                loss = loss.mean()
                epoch_loss += loss
                loss.backward()
                self.optimizer.step()
                print(epoch,step,loss)
                self.all_step += 1
                self.writer.add_scalar('data/loss_step', loss, self.all_step)
            averge_loss = epoch_loss/self.data_quantity
            epoch += 1
            self.writer.add_scalar('data/loss_group', {'train': averge_loss}, averge_loss, epoch)
            if epoch % 5 == 0 :
                self.net.eval()
                for step,sample in (enumerate(self.valloader)):
                    xs = [x.cuda(non_blocking=True) for x in sample['xs']]
                    ys = [x.cuda(non_blocking=True) for x in sample['ys']]
                    out = self.net._train(xs)
                    loss = self.loss(out,ys)
                    loss = loss.mean()
                    epoch_loss += loss
                    print(epoch,step,loss)
                averge_loss = epoch_loss/len(self.valloader)
                self.writer.add_scalar('data/loss_group', {'val': averge_loss}, averge_loss, epoch)
                self.net.train()
            if (epoch % self.snapshot) == self.snapshot - 1:
                torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'cornernet' + '_epoch-' + str(epoch) + '.pth'))
                print("Save model at {}\n".format(os.path.join(self.save_dir, 'cornernet' + '_epoch-' + str(epoch) + '.pth')))
            if self.all_step >= self.stepsize:
                sys.exit('Finish training')

if __name__=="__main__":
    t=Train()
    t.train_single()
