from __future__ import print_function
import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import triplet.siamese_net as models
from triplet.tripletnet import Tripletnet
from triplet.utils import *

import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import random


class ReidNetwork():
    """
    checkpoint: 'latest' (epoch 16) or 'best' (epoch 8)
    """
    def __init__(self, checkpoint):

        torch.cuda.manual_seed(1)

        self.transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                        ])

        self.checkpoint = checkpoint
        self.net = self.load_net().cuda()
        self.net.eval()

    def load_net(self):
        embedding_model = models.vgg.vgg16(pretrained=True)
        tnet = Tripletnet(embedding_model, opts=None)

        if self.checkpoint == 'latest':
            tnet.load_state_dict(torch.load('triplet/out/TripletNet/checkpoint.pth.tar')['state_dict'])
            return tnet
        elif self.checkpoint == 'best':
            tnet.load_state_dict(torch.load('triplet/out/model_best.pth.tar')['state_dict'])
            return tnet
        else:
            raise Exception("No checkpoint model")

    def compute_score(self, img1_path, img2_path, bbox1, bbox2):
        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)
        print(bbox1)

        img1 = img1[bbox1[1]:bbox1[3],bbox1[0]:bbox1[2],:]
        img2 = img2[bbox2[1]:bbox2[3],bbox2[0]:bbox2[2],:]

        img1 = Image.fromarray(img1, 'RGB')
        img2 = Image.fromarray(img2, 'RGB')

        img1 = self.transform(img1).unsqueeze(0)
        img2 = self.transform(img2).unsqueeze(0)

        img1, img2 = Variable(img1.cuda()), Variable(img2.cuda())

        dista, distb, _, _, _ = self.net(img1, img2, img2)

        return dista.cpu().data


    def plot_pic(self, img1_path, img2_path, bbox1, bbox2):
        img1 = io.imread(img1_path)
        img2 = io.imread(img2_path)

        img1 = img1[bbox1[1]:bbox1[3],bbox1[0]:bbox1[2],:]
        img2 = img2[bbox2[1]:bbox2[3],bbox2[0]:bbox2[2],:]

        plt.figure()
        plt.title('display')
        plt.subplot(121)
        plt.imshow(img1)
        plt.subplot(122)
        plt.imshow(img2)
        plt.show()        

if __name__ == '__main__':
    Reid = ReidNetwork('latest')

    davis_path = '/home/shuangjie/shuangjie/daizong/DAVIS2017/test_dev'
    seqname = 'carousel'
    img_num = len(os.listdir(os.path.join(davis_path, 'JPEGImages', '480p', seqname)))

    '''

    img_id1 = random.randint(0,img_num-1)
    img_id2 = random.randint(0,img_num-1)

    obj1 = 1
    obj2 = 3

    img1_path = os.path.join(davis_path, 'JPEGImages', '480p', seqname, str(img_id1).zfill(5)+'.jpg')
    img2_path = os.path.join(davis_path, 'JPEGImages', '480p', seqname, str(img_id2).zfill(5)+'.jpg')
    img3_path = os.path.join(davis_path, 'JPEGImages', '480p', seqname, str(img_id1).zfill(5)+'.jpg')

    img1 = io.imread(img1_path)
    mask1,_ = load_mask(os.path.join(davis_path, 'Annotations', '480p', seqname, str(img_id1).zfill(5)+'.png'), obj1)
    bbox1 = compute_bbox_from_mask(mask1)

    img2 = io.imread(img2_path)
    mask2,_ = load_mask(os.path.join(davis_path, 'Annotations', '480p', seqname, str(img_id2).zfill(5)+'.png'), obj1)
    bbox2 = compute_bbox_from_mask(mask2)

    img3 = io.imread(img3_path)
    mask3,_ = load_mask(os.path.join(davis_path, 'Annotations', '480p', seqname, str(img_id1).zfill(5)+'.png'), obj2)
    bbox3 = compute_bbox_from_mask(mask3)

    Reid.plot_pic(img1_path, img2_path, bbox1, bbox2)
    score_close = Reid.compute_score(img1_path, img2_path, bbox1, bbox2)
    Reid.plot_pic(img1_path, img3_path, bbox1, bbox3)
    score_far = Reid.compute_score(img1_path, img3_path, bbox1, bbox3)
    print(score_close)
    print(score_far)
    '''
    img1_path = os.path.join(davis_path, 'JPEGImages', '480p', seqname, str(0).zfill(5)+'.jpg')
    img2_path = os.path.join(davis_path, 'JPEGImages', '480p', seqname, str(3).zfill(5)+'.jpg')#5#48#40#23
    bbox1 = [27, 223, 419, 474]
    bbox2 = [533, 130, 795, 379]#[116,232,566,477]## [200,212,645,477]#[2,162,174,407]#[124,122,339,276]#[453,87,652,273]
    Reid.plot_pic(img1_path, img2_path, bbox1, bbox2)
    score_close = Reid.compute_score(img1_path, img2_path, bbox1, bbox2)
    print(score_close)

