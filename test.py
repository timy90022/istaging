# import tensorflow as tf
import os
from utils.get_data import Image_data
from models import CornerNet
from config import cfg
import numpy as np
import cv2
import torch
# from .kp_utils import _sigmoid
# import timem    
from external.nms import soft_nms, soft_nms_merge

from dataloaders import istg
from torch.utils.data import DataLoader
# from test import Debug
class Train():
    def __init__(self):
        # os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.net=CornerNet.model()
        self.net.cuda()
        self.net.eval()

        self.net.load_state_dict(torch.load('./check/cornernet_epoch-119.pth'))
        
        voc_train = istg.Segmentation(split='test')

        self.trainloader = DataLoader(voc_train, batch_size=1, shuffle=False, num_workers=1,drop_last=True)
        self.gpus=[0]
        self.score = True
    def write_file(self,file,box,now_map):
        if self.score==True:
            if now_map==0:
                file.write('%d %f %d %d %d %d\n'%(box[7]/4,box[4]/4,box[0],box[1],box[2],box[3]))
            else:
                box[0],box[2] = box[0]+512,box[2]+512
                if box[0] >1024 and box[2]>1024:
                    box[0] = box[0]-1024
                    box[2] = box[2]-1024
                    file.write('%d %f %d %d %d %d\n'%(box[7]/4,box[4]/4,box[0],box[1],box[2],box[3]))
                elif box[2]>1024:
                    box[2] = box[2]-1024
                    file.write('%d %f %d %d %d %d\n'%(box[7]/4,box[4]/4,box[0],box[1],1024,box[3]))
                    file.write('%d %f %d %d %d %d\n'%(box[7]/4,box[4]/4,0,box[1],box[2],box[3]))
        else:
            if now_map==0:
                file.write('%d %f %d %d %d %d\n'%(box[7]/4,box[4]/4,box[0],box[1],box[2],box[3]))
            else:
                box[0],box[2] = box[0]+512,box[2]+512
                box[0] = box[0]-1024 if box[0]>1024 else box[0]
                box[2] = box[2]-1024 if box[2]>1024 else box[2]
                file.write('%d %f %d %d %d %d\n'%(box[7]/4,box[4]/4,box[0],box[1],box[2],box[3]))
        return file



    def train_single(self):

        for epoch in range(1):
            for step,(sample1,sample2) in enumerate(self.trainloader):
                print(sample1['name'][0])
    
                xs1 = [x.cuda(non_blocking=True) for x in sample1['xs']]
                xs2 = [x.cuda(non_blocking=True) for x in sample2['xs']]
                # ys = [x.cuda(non_blocking=True) for x in sample['ys']]
                
                
                # out1 = self.net._test(xs1, ae_threshold=0.5, K=100, kernel=3)
                # out2 = self.net._test(xs2, ae_threshold=0.5, K=100, kernel=3)
                out1 = self.net._test(xs1, kernel=3, ae_threshold=0.5)
                out2 = self.net._test(xs2, kernel=3, ae_threshold=0.5)
                file = open('predicted/' + sample1['name'][0] + '.txt','w')



                rgb = xs1[0].cpu().numpy()
                rgb = ((rgb[0,::]+1)/2*255).transpose((1,2,0))
                rgb = np.array(rgb).astype(np.uint8).copy()
                blank = np.zeros((512,1024))

                out1 = out1[0].cpu().detach().numpy()*4
                out1 = np.hstack((out1,np.zeros((1000, 1))))

                out2 = out2[0].cpu().detach().numpy()*4
                out2 = np.hstack((out2,np.ones((1000, 1))))


                out = np.concatenate((out1,out2), axis=0)
                small = np.where(out[:,4]<0.35)
                out = np.delete(out,small[0],0)
                sequence = np.argsort(-out[:,4])


                now_map = 0
                for i in sequence:
                    box = out[i,:]
                    # assert False
                    dif = (box[2]-box[0])/(box[3]-box[1])
                    area = (box[2]-box[0])*(box[3]-box[1])
                    
                    if box[-1]!=now_map:
                        rgb = np.concatenate((rgb[:, 512:,:], rgb[:, :512,:]), axis=1)
                        blank = np.concatenate((blank[:, 512:], blank[:, :512]), axis=1)
                        now_map = 1-now_map

                    crop = blank[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                    if ( 1 not in np.unique(crop)):
                        if box[-2]/4==1 and  dif<4 and area<64000 and box[4]>2:
                            cv2.rectangle(rgb,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),1)
                            cv2.putText(rgb, str(round(box[4], 2)), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 255, 255), 1)
                            blank[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = 1
                            self.write_file(file,box,now_map)
                        elif box[-2]/4==0 and dif<1.5 and box[4]>0.5:
                            cv2.rectangle(rgb,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),1)
                            cv2.putText(rgb, str(round(box[4], 2)), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 255, 255), 1)
                            blank[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = 1
                            self.write_file(file,box,now_map)

                    cv2.imwrite('rgb/' + sample1['name'][0] + '.jpg',rgb)
                file.close()
                if 0!=now_map:
                    rgb = np.concatenate((rgb[:, 512:,:], rgb[:, :512,:]), axis=1)
                cv2.imwrite('rgb/' + sample1['name'][0] + '.jpg',rgb)
                    # cv2.imwrite('test.jpg',rgb)
                # assert False
                



if __name__=="__main__":

    t=Train()
    t.train_single()
