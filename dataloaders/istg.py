from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
# from mypath import Path
import math
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import glob
import cv2
import random
from dataloaders import custom_transforms as tr
from torchvision import transforms

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def normalize_(image, mean, std):
    image -= mean
    image /= std

class Segmentation(Dataset):
    """
    Nyu dataset
    """

    def __init__(self,
                 base_dir='/home/timy90022/dataset/istg/',
                 split='train'
                 ):
        """
        :param split: train/val/test
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self.rotate = True
        self.gaussian_bump = True
        self.gaussian_rad = -1
        self.gaussian_iou = 0.7
        self.split = split
        self.max_ratio = 1
        self.min_ratio = 1
        self.record_ratio_door = []
        self.record_ratio_window = []
        self.record_area_door = []
        self.record_area_window = []


        self.mean = (0.5,0.5,0.5)
        self.std = (0.5,0.5,0.5)
        self.transform = composed_transforms_tr = transforms.Compose([
                            tr.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                            ])
        self.output_size = [128,256]
        self.input_size = [512,1024]
        self.width_ratio  = self.output_size[1] / self.input_size[1]
        self.height_ratio = self.output_size[0] / self.input_size[0]
        

        self.all_image = []
        self.all_label = []
        self.all_assist = []

        first = sorted(glob.glob(self._base_dir + '/data_obj2d/istg' + '/*'))
        for i in first:
            second = sorted(glob.glob(i + '/*'))
            for j in second:
                # print(j)
                file_path_one = j + '/color.png'
                file_path_two = j + '/obj2d.png'
                file_path_three = j.replace('data_obj2d','fcmaps') + '/fcmap.png'
                
                # print(os.path.isfile(file_path_two))
                # print(file_path_two)
                if ( os.path.isfile(file_path_one) and os.path.isfile(file_path_two) and os.path.isfile(file_path_three)):
                    self.all_image.append(file_path_one)
                    self.all_label.append(file_path_two)
                    self.all_assist.append(file_path_three)

        first = sorted(glob.glob(self._base_dir + '/data_obj2d/sun360' + '/*'))
        for i in first:
            # print(j)
            file_path_one = i + '/color.png'
            file_path_two = i + '/obj2d.png'
            file_path_three = i.replace('data_obj2d','fcmaps') + '/fcmap.png'
            # print(os.path.isfile(file_path_two))
            # print(file_path_two)
            if ( os.path.isfile(file_path_one) and os.path.isfile(file_path_two) and os.path.isfile(file_path_three)):
                self.all_image.append(file_path_one)
                self.all_label.append(file_path_two)
                self.all_assist.append(file_path_three)

        
        assert (len(self.all_image) == len(self.all_label))

        if self.split =='train':
            stay = [0,1,2,3,4,5,6,7,8]
        else:
            stay = [9]
        self.all_image = [self.all_image[e] for e in range(len(self.all_image)) if e%10 in stay]
        self.all_label = [self.all_label[e] for e in range(len(self.all_label)) if e%10 in stay]
        self.all_assist = [self.all_assist[e] for e in range(len(self.all_assist)) if e%10 in stay]

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.all_image)))

    def __len__(self):
        
        return len(self.all_image)

    def __getitem__(self, index):
        # if index == len(self.all_image)-1:
        #     print(self.record_ratio_door)
        #     plt.hist(self.record_ratio_door,300)
        #     plt.savefig("door.png")
        #     plt.close('all')
        #     plt.hist(self.record_ratio_window,300)
        #     plt.savefig("window.png")
        #     plt.close('all')

        #     plt.hist(self.record_area_door,300)
        #     plt.savefig("door_area.png")
        #     plt.close('all')
        #     plt.hist(self.record_area_window,300)
        #     plt.savefig("window_area.png")

        #     assert False
        if self.split =='train' or self.split =='val':
            return self._make_img_gt_point_pair(index)
        else:
            return self.test_pair(index,0,True),self.test_pair(index,512,False)

    def _make_img_gt_point_pair(self, index):
        name =  self.all_image[index].split('/')[-3] + '-' + self.all_image[index].split('/')[-2]

        max_tag_len = 128
        # image       = np.zeros((self.input_size[0], self.input_size[1],3), dtype=np.float32)
        heatmaps_tl = np.zeros((self.output_size[0], self.output_size[1],2), dtype=np.float32)
        heatmaps_br = np.zeros((self.output_size[0], self.output_size[1],2), dtype=np.float32)
        offsets_tl    = np.zeros((max_tag_len, 2), dtype=np.float32)
        offsets_br    = np.zeros((max_tag_len, 2), dtype=np.float32)
        tags_tl     = np.zeros((max_tag_len), dtype=np.int64)
        tags_br     = np.zeros((max_tag_len), dtype=np.int64)
        tags_mask   = np.zeros((max_tag_len), dtype=np.uint8)
        boxes       = np.zeros((max_tag_len,4), dtype=np.int64)
        ratio       = np.ones((max_tag_len,2), dtype=np.float32)
        tag_lens    = 0

        # read image
        _img = cv2.imread(self.all_image[index])
        _img=cv2.resize(_img,(1024,512),interpolation=cv2.INTER_CUBIC)
        _target = cv2.imread(self.all_label[index])
        assist = cv2.imread(self.all_assist[index])

        # flip image
        flip = random.randint(0,1)
        if flip==1 and self.rotate:
            _img = cv2.flip(_img, 1)
            _target = cv2.flip(_target, 1)
            assist = cv2.flip(assist, 1)

        # displacement
        tmp = np.sum(_target[...,0],axis=0)
        tmp = np.where(tmp==0)[0]
        dis = random.choice(tmp)

        _img = np.concatenate((_img[:, dis:,:], _img[:, :dis,:]), axis=1)  # 右
        _img = self.transform(_img)
        _target = np.concatenate(( _target[:, dis:,:],_target[:, :dis,:]), axis=1)  # 右
        assist = np.concatenate((assist[:, dis:,:], assist[:, :dis,:]), axis=1)  # 右
        

        component_t = cv2.cvtColor(_target, cv2.COLOR_BGR2GRAY)       
        component_t = cv2.threshold(component_t, 127, 255, cv2.THRESH_BINARY)[1]
        ret_t, labels_t = cv2.connectedComponents(component_t)


        component_a = cv2.cvtColor(assist, cv2.COLOR_BGR2GRAY)
        component_a = cv2.threshold(component_a, 127, 255, cv2.THRESH_BINARY)[1]
        ret_a, labels_a = cv2.connectedComponents(component_a)

        classification = []
        object_p = np.unique(labels_t).astype(int)
        for i in object_p:
            if i != 0:
                coordinate = np.where(labels_t==i)
                xtl_ori = np.amin(coordinate[1])
                xbr_ori = np.amax(coordinate[1])

                ytl_ori = coordinate[0][np.where(coordinate[1]==xtl_ori)[0][0]]
                ybr_ori = coordinate[0][np.where(coordinate[1]==xbr_ori)[0][-1]]

                p_index = np.argmax(coordinate[0])
                x, y = coordinate[1][p_index],coordinate[0][p_index]
                tmp = np.amin(np.where(labels_a[:,x]==2)[0])
                distance = abs(tmp-y)

                if distance<5:
                    category = 0
                else:
                    category = 1


                fxtl = (xtl_ori * self.width_ratio)
                fytl = (ytl_ori * self.height_ratio)
                fxbr = (xbr_ori * self.width_ratio)
                fybr = (ybr_ori * self.height_ratio)

                xtl = int(fxtl)
                ytl = int(fytl)
                xbr = int(fxbr)
                ybr = int(fybr)

                if self.gaussian_bump:
                    width  = xbr_ori - xtl_ori
                    height = ybr_ori - ytl_ori

                    width  = math.ceil(width * self.width_ratio)
                    height = math.ceil(height * self.height_ratio)

                    if self.gaussian_rad == -1:
                        radius = gaussian_radius((height, width), self.gaussian_iou)
                        radius = max(0, int(radius))
                    else:
                        radius = self.gaussian_rad

                    draw_gaussian(heatmaps_tl[:,:,category], [xtl, ytl], radius)
                    draw_gaussian(heatmaps_br[:,:,category], [xbr, ybr], radius)
                else:
                    heatmaps_tl[ytl, xtl, category] = 1
                    heatmaps_br[ybr, xbr, category] = 1

                tag_ind = tag_lens
                offsets_tl[tag_ind, :] = [fxtl - xtl, fytl - ytl]
                offsets_br[tag_ind, :] = [fxbr - xbr, fybr - ybr]
                tags_tl[tag_ind] = ytl * self.output_size[1] + xtl
                tags_br[tag_ind] = ybr * self.output_size[1] + xbr
                boxes[tag_ind] = [xtl_ori,ytl_ori,xbr_ori,ybr_ori]
                ratio[tag_ind] = [self.width_ratio,self.height_ratio]
                tag_lens += 1

        tags_mask[:tag_lens] = 1

        images      = torch.from_numpy(_img.transpose((2, 0, 1)))
        tl_heatmaps = torch.from_numpy(heatmaps_tl.transpose((2, 0, 1)))
        br_heatmaps = torch.from_numpy(heatmaps_br.transpose((2, 0, 1)))
        tl_regrs    = torch.from_numpy(offsets_tl)
        br_regrs    = torch.from_numpy(offsets_br)
        tl_tags     = torch.from_numpy(tags_tl)
        br_tags     = torch.from_numpy(tags_br)
        tag_masks   = torch.from_numpy(tags_mask)

        return {
        "xs": [images, tl_tags, br_tags],
        "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs],
        "name" : name
    }

    def test_pair(self, index,dis,save):


        name =  self.all_image[index].split('/')[-3] + '-' + self.all_image[index].split('/')[-2]
        if save:
            file = open('ground-truth/' + str(name) + '.txt','w')

        max_tag_len = 128
        # image       = np.zeros((self.input_size[0], self.input_size[1],3), dtype=np.float32)
        heatmaps_tl = np.zeros((self.output_size[0], self.output_size[1],2), dtype=np.float32)
        heatmaps_br = np.zeros((self.output_size[0], self.output_size[1],2), dtype=np.float32)
        offsets_tl    = np.zeros((max_tag_len, 2), dtype=np.float32)
        offsets_br    = np.zeros((max_tag_len, 2), dtype=np.float32)
        tags_tl     = np.zeros((max_tag_len), dtype=np.int64)
        tags_br     = np.zeros((max_tag_len), dtype=np.int64)
        tags_mask   = np.zeros((max_tag_len), dtype=np.uint8)
        boxes       = np.zeros((max_tag_len,4), dtype=np.int64)
        ratio       = np.ones((max_tag_len,2), dtype=np.float32)
        tag_lens    = 0

        # read image
        _img = cv2.imread(self.all_image[index])
        _img=cv2.resize(_img,(1024,512),interpolation=cv2.INTER_CUBIC)
        _target = cv2.imread(self.all_label[index])
        assist = cv2.imread(self.all_assist[index])

        # displacement
        tmp = np.sum(_target[...,0],axis=0)
        tmp = np.where(tmp==0)[0]
        # dis = tmp[0] + dis
        _img = np.concatenate((_img[:, dis:,:], _img[:, :dis,:]), axis=1)  # 右
        _img = self.transform(_img)
        _target = np.concatenate(( _target[:, dis:,:],_target[:, :dis,:]), axis=1)  # 右
        assist = np.concatenate((assist[:, dis:,:], assist[:, :dis,:]), axis=1)  # 右
        

        component_t = cv2.cvtColor(_target, cv2.COLOR_BGR2GRAY)       
        component_t = cv2.threshold(component_t, 127, 255, cv2.THRESH_BINARY)[1]
        ret_t, labels_t = cv2.connectedComponents(component_t)


        component_a = cv2.cvtColor(assist, cv2.COLOR_BGR2GRAY)
        component_a = cv2.threshold(component_a, 127, 255, cv2.THRESH_BINARY)[1]
        ret_a, labels_a = cv2.connectedComponents(component_a)

        classification = []
        object_p = np.unique(labels_t).astype(int)
        for i in object_p:
            if i != 0:
                coordinate = np.where(labels_t==i)
                xtl_ori = np.amin(coordinate[1])
                xbr_ori = np.amax(coordinate[1])

                ytl_ori = coordinate[0][np.where(coordinate[1]==xtl_ori)[0][0]]
                ybr_ori = coordinate[0][np.where(coordinate[1]==xbr_ori)[0][-1]]


                p_index = np.argmax(coordinate[0])
                x, y = coordinate[1][p_index],coordinate[0][p_index]
                tmp = np.amin(np.where(labels_a[:,x]==2)[0])
                distance = abs(tmp-y)

                if distance<5:
                    category = 0
                else:
                    category = 1
                if save:
                    file.write('%d %d %d %d %d\n'%(category,xtl_ori,ytl_ori,xbr_ori,ybr_ori))


                fxtl = (xtl_ori * self.width_ratio)
                fytl = (ytl_ori * self.height_ratio)
                fxbr = (xbr_ori * self.width_ratio)
                fybr = (ybr_ori * self.height_ratio)
                
                try:
                    dif = (fxbr-fxtl)/(fybr-fytl)
                    area = (fxbr-fxtl)*(fybr-fytl)
                    if category == 0:
                        self.record_ratio_door.append(dif)
                        self.record_area_door.append(area)
                    else:
                        self.record_ratio_window.append(dif)
                        self.record_area_window.append(area)
                        
                    # if dif>self.max_ratio:
                    #     self.max_ratio = dif
                    #     print(self.min_ratio,self.max_ratio)
                    #     cv2.imwrite('tmp/'+str(dif)+'.jpg',_target)
                    # elif dif<self.min_ratio:
                    #     self.min_ratio = dif
                    #     print(self.min_ratio,self.max_ratio)
                    #     cv2.imwrite('tmp/'+str(dif)+'.jpg',_target)
                except:
                    pass
                
                
                




                xtl = int(fxtl)
                ytl = int(fytl)
                xbr = int(fxbr)
                ybr = int(fybr)

                if self.gaussian_bump:
                    width  = xbr_ori - xtl_ori
                    height = ybr_ori - ytl_ori

                    width  = math.ceil(width * self.width_ratio)
                    height = math.ceil(height * self.height_ratio)

                    if self.gaussian_rad == -1:
                        radius = gaussian_radius((height, width), self.gaussian_iou)
                        radius = max(0, int(radius))
                    else:
                        radius = self.gaussian_rad

                    draw_gaussian(heatmaps_tl[:,:,category], [xtl, ytl], radius)
                    draw_gaussian(heatmaps_br[:,:,category], [xbr, ybr], radius)
                else:
                    heatmaps_tl[ytl, xtl, category] = 1
                    heatmaps_br[ybr, xbr, category] = 1

                tag_ind = tag_lens
                offsets_tl[tag_ind, :] = [fxtl - xtl, fytl - ytl]
                offsets_br[tag_ind, :] = [fxbr - xbr, fybr - ybr]
                tags_tl[tag_ind] = ytl * self.output_size[1] + xtl
                tags_br[tag_ind] = ybr * self.output_size[1] + xbr
                boxes[tag_ind] = [xtl_ori,ytl_ori,xbr_ori,ybr_ori]
                ratio[tag_ind] = [self.width_ratio,self.height_ratio]
                tag_lens += 1

        tags_mask[:tag_lens] = 1

        images      = torch.from_numpy(_img.transpose((2, 0, 1)))
        tl_heatmaps = torch.from_numpy(heatmaps_tl.transpose((2, 0, 1)))
        br_heatmaps = torch.from_numpy(heatmaps_br.transpose((2, 0, 1)))
        tl_regrs    = torch.from_numpy(offsets_tl)
        br_regrs    = torch.from_numpy(offsets_br)
        tl_tags     = torch.from_numpy(tags_tl)
        br_tags     = torch.from_numpy(tags_br)
        tag_masks   = torch.from_numpy(tags_mask)
        if save:   
            file.close()

        return {
        "xs": [images, tl_tags, br_tags],
        "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs],
        "name" : name
    }



