# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
from skimage import io
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from dataloaders.coco_transforms import *
# from coco_transforms import *

class COCODataset(Dataset):
    def __init__(self, dataset_name, rootdir, period):
        self.dataset_name = dataset_name
        self.root_dir = rootdir
        self.dataset_dir = self.root_dir
        
        self.period = period
        self.year = self.__get_year()
        self.img_dir = os.path.join(self.dataset_dir,'%s%s'%(self.period,self.year))
        self.ann_dir = os.path.join(self.dataset_dir, 'annotations/instances_%s%s.json'%(self.period,self.year))
        self.rescale = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomhsv = None
        self.totensor = ToTensor()
    
        self.coco = COCO(self.ann_dir)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.imgIds = self.coco.getImgIds()
        self.catIds = self.coco.getCatIds()

        # if cfg.DATA_RESCALE > 0:
        self.rescale = Rescale(512)
        if self.period == 'train':        
            # if cfg.DATA_RANDOMCROP > 0:
            self.randomcrop = RandomCrop(512)
            # if cfg.DATA_RANDOMROTATION > 0 or cfg.DATA_RANDOMSCALE != 1:
            #     self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION,cfg.DATA_RANDOMSCALE)
            # if cfg.DATA_RANDOMFLIP > 0:
            self.randomflip = RandomFlip(0.5)
            # if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
            self.randomhsv = RandomHSV(10, 30, 30)
        # self.cfg = cfg
    
    def __get_year(self):
        name = self.dataset_name
        if 'coco' in name:
            name = name.replace('coco','')
        else:
            name = name.replace('COCO','')
        year = name
        return year

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img_ann = self.coco.loadImgs(self.imgIds[idx])
        name = os.path.join(self.img_dir, img_ann[0]['file_name'])
        image = cv2.imread(name)
        # cv2.imshow('1', image)
        # cv2.waitKey(0)
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r,c,_ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        
        if self.period == 'train':
            annIds = self.coco.getAnnIds(imgIds=self.imgIds[idx])
            anns = self.coco.loadAnns(annIds)
            # print(len(anns))
            segmentation = np.zeros((r,c),dtype=np.uint8)
            for ann_item in anns:
                mask = self.coco.annToMask(ann_item)
                segmentation[mask>0] = ann_item['category_id']
            if np.max(segmentation)>91:
                print(np.max(segmentation))
                raise ValueError('segmentation > 91')
            sample['segmentation'] = segmentation

            # if self.cfg.DATA_RANDOM_H > 0 or self.cfg.DATA_RANDOM_S > 0 or self.cfg.DATA_RANDOM_V > 0:
            sample = self.randomhsv(sample)
            # if self.cfg.DATA_RANDOMFLIP > 0:
            sample = self.randomflip(sample)
            # if self.cfg.DATA_RANDOMROTATION > 0 or self.cfg.DATA_RANDOMSCALE != 1:
            #     sample = self.randomrotation(sample)
            # if self.cfg.DATA_RANDOMCROP > 0:
            sample = self.randomcrop(sample)

        # if self.cfg.DATA_RESCALE > 0:
        sample = self.rescale(sample)
        if 'segmentation' in sample.keys():
            sample['segmentation_onehot'] = onehot(sample['segmentation'], 91)
        sample = self.totensor(sample)

        sample['image'] = sample['image'].unsqueeze(0)
        sample['segmentation'] = sample['segmentation'].unsqueeze(0)
        sample['image'] = torch.cat([sample['image'], sample['image']], 0)
        sample['segmentation'] = torch.cat([sample['segmentation'], sample['segmentation']], 0)
        # print(sample['image'].shape)
        return sample
 
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3 | (m&64)>>1
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2 | (m&128)>>2
        cmap[:,:,2] = (m&4)<<5 | (m&32)<<1
        return cmap

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    train_data = COCODataset(dataset_name='coco2014', rootdir='../../cocodata', period='train')
    dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
    for ii, sample in enumerate(dataloader):
        img_tmp, segmap = sample['image'], sample['segmentation']
        img_tmp = img_tmp.squeeze(0)
        segmap = segmap.squeeze(0)
        print(img_tmp.shape)
        print(segmap.shape)
        plt.figure()
        plt.title('display')
        plt.subplot(211)
        plt.imshow(np.transpose(np.array(img_tmp), (1,2,0)))
        plt.subplot(212)
        plt.imshow(segmap)
        break
    plt.show()