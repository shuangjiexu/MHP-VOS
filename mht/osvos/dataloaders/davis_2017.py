from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize
import json

import torch
# from helpers import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .utils import *
# from utils import *

from PIL import Image
from matplotlib import pyplot as plt

# def imread_indexed(filename):
#     """ Load image given filename."""

#     im = Image.open(filename)

#     annotation = np.atleast_3d(im)[...,0]
#     return annotation,np.array(im.getpalette()).reshape((-1,3))

# def read_flow(flowfile):
#     f = open(flowfile, 'rb')
#     header = f.read(4)
#     if header.decode("utf-8") != 'PIEH':
#         raise Exception('Flow file header does not contain PIEH')

#     width = np.fromfile(f, np.int32, 1).squeeze()
#     height = np.fromfile(f, np.int32, 1).squeeze()

#     flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
#     # print(flow.max())
#     return flow.astype(np.float32)

# cv2.setNumThreads(0)

class OfflineDataset(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='../multi_mask',
                 flow_root_dir='../trainval_flow',
                 train_val_file = '../DAVIS2017/trainval/ImageSets/2017',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 obj_id=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.flow_root_dir = flow_root_dir
        self.transform = transform
        self.meanval = meanval
        self.obj_id = obj_id

        if self.train:
            fname = 'train'
        else:
            fname = 'val'


        # Initialize the original DAVIS splits for training the parent network
        # with open(os.path.join(train_val_file, fname + '.txt')) as f:
        #     seqs = f.readlines()
        #     img_list = []
        #     labels = []
        #     flow_list = []
        #     former_mask = []
        #     for seq in seqs:
        #         images = np.sort(os.listdir(os.path.join(db_root_dir, seq.strip(), 'origin')))
        #         images_path = list(map(lambda x: os.path.join(seq.strip(), 'origin', x), images))
        #         flows = np.sort(os.listdir(os.path.join(flow_root_dir, seq.strip())))
        #         flows_path = list(map(lambda x: os.path.join(seq.strip(), x), flows))

        #         obj_num = len(os.listdir(os.path.join(db_root_dir, seq.strip()))) - 1
        #         for obj_index in range(obj_num):
        #             for img_num in range(len(images_path)-1):
        #                 check_mask = get_obj_num(os.path.join(self.db_root_dir, seq.strip(), str(obj_index+1), str(img_num+1).zfill(5)+'.png'))
        #                 if check_mask == 1:
        #                     continue
        #                 if seq.strip() == 'lindy-hop' and obj_index == 0 and img_num == 69:
        #                     continue
        #                 area_mask,_ = imread_indexed(os.path.join(self.db_root_dir, seq.strip(), str(obj_index+1), str(img_num+1).zfill(5)+'.png'))
        #                 mask_area = valid_area(area_mask)
        #                 if mask_area < 20*20:
        #                     continue

        #                 img_list.append(images_path[img_num+1])
        #                 # flow_list.append(flows_path[img_num])
        #                 lab = np.sort(os.listdir(os.path.join(db_root_dir, seq.strip(), str(obj_index+1))))
        #                 lab_path = list(map(lambda x: os.path.join(seq.strip(), str(obj_index+1), x), lab))
        #                 labels.append(lab_path[img_num+1])

        #                 check_mask_f = get_obj_num(os.path.join(self.db_root_dir, seq.strip(), str(obj_index+1), str(img_num).zfill(5)+'.png'))
        #                 if check_mask_f == 1:
        #                     former_mask_f = lab_path[img_num+1]
        #                 else:
        #                     lab_f = np.sort(os.listdir(os.path.join(db_root_dir, seq.strip(), str(obj_index+1))))
        #                     lab_f_path = list(map(lambda x: os.path.join(seq.strip(), str(obj_index+1), x), lab_f))
        #                     former_mask_f = lab_f_path[img_num]
        #                 former_mask.append(former_mask_f)
        #                 # if lab_path[img_num+1].split('/')[-1] == former_mask_f.split('/')[-1]:
        #                 #     print(former_mask_f)
        #                 # print(images_path[img_num+1])
        #                 # print(lab_path[img_num+1])
        #                 # print(former_mask_f)
        # print(len(img_list))
        # if len(img_list)%2 != 0:
        #     img_list.append(images_path[img_num+1])
        #     # flow_list.append(flows_path[img_num])
        #     labels.append(lab_path[img_num+1])
        #     former_mask.append(former_mask_f)
        # data = {}
        # data['img_list'] = img_list
        # data['labels'] = labels
        # data['former_mask'] = former_mask
        # with open('data_val.json', 'w') as f:
        #     json.dump(data, f)
        if self.train:
            json_file = './dataloaders/data_train.json'
        else:
            json_file = './dataloaders/data_val.json'

        with open(json_file, 'r') as f:
            data = json.load(f)

        img_list = data['img_list']
        labels = data['labels']
        former_mask = data['former_mask']

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        # self.flow_list = flow_list
        self.former_mask = former_mask

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'label': gt}

        # if self.seq_name is not None:
        #     fname = os.path.join(self.seq_name, "%05d" % idx)
        #     sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)
        # plt.figure()
        # plt.title('display')
        # plt.subplot(411)
        # plt.imshow(Image.fromarray(np.uint8(img)))
        # plt.subplot(412)
        # plt.imshow(gt)
        # plt.subplot(413)
        # plt.imshow(mask[:,:,0])
        # plt.subplot(414)
        # plt.imshow(flow[:,:,0])
        # plt.show()
        # print(sample['image'].shape)
        # print(sample['mask'].shape)
        # bb

        # print(sample['image'].shape)
        # print(sample['mask'].shape)
        # print(sample['flow'].shape)
        # print(sample['label'].shape)
        # bb
        # sample['image'] = torch.cat([sample['image'], sample['mask'], sample['flow']], 0)
        # sample.pop('mask')
        # sample.pop('flow')

        # if self.train:
        #     sample['image'] = sample['image'].unsqueeze(0)
        #     sample['label'] = sample['label'].unsqueeze(0)
        #     sample['image'] = torch.cat([sample['image'], sample['image']], 0)
        #     sample['label'] = torch.cat([sample['label'], sample['label']], 0)
        # print(sample['image'].shape)
        # bb
        # self.get_img_filename(idx)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        # idx=13

        label_ ,_ = imread_indexed(os.path.join(self.db_root_dir, self.labels[idx]))
        label = label_.copy()
        gt = np.array(label, dtype=np.float32)

        bbox = compute_bbox_from_mask(gt)
        gt = gt[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)

        # img = Image.open(os.path.join(self.db_root_dir, self.img_list[idx]))
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        img = cv2.resize(img, (512,512))
        img = Image.fromarray(img)
        img = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(img)
        # img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        # print(img.min())

        f_mask_ ,_ = imread_indexed(os.path.join(self.db_root_dir, self.former_mask[idx]))
        f_mask = f_mask_.copy()
        mask = np.array(f_mask, dtype=np.float32)

        im2_id = int(self.img_list[idx].split('/')[-1].split('.')[0])
        im1_id = im2_id - 1
        obj_id = int(self.labels[idx].split('/')[1])

        flow_dir = os.path.join(self.flow_root_dir, self.img_list[idx].split('/')[0])
        img_dir = os.path.join(self.db_root_dir, self.img_list[idx].split('/')[0], 'origin')
        warped_mask, validflowmap01,_,_ = warp_mask(mask, im1_id, im2_id, flow_dir, img_dir)
        warped_mask = (warped_mask > 0.3).astype(np.float32)
        warped_mask = warped_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        warped_mask = cv2.resize(warped_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        data = np.concatenate((img,np.expand_dims(warped_mask, axis=2)), axis=2)
        # print(warped_mask.max())
        # warped_mask = np.subtract(warped_mask, np.array(self.meanval, dtype=np.float32))
        # print(warped_mask)
        # print(warped_mask.min())
        # print(warped_mask.shape)
        # print(obj_id)      
        # print(self.img_list[idx])
        # print(self.labels[idx])
        # print(self.former_mask[idx])

        
        # # print(bbox)
        # # print(self.img_list[idx])
        # # print(self.labels[idx])

        # mask = mask*255
        # # print(mask.max())
        # # print(mask.min())
        # mask = np.expand_dims(mask, axis=2)

        # flow = read_flow(os.path.join(self.flow_root_dir, self.flow_list[idx]))
        # flow[:,:,1] = (flow[:,:,1] - flow[:,:,1].min()) *255 / (flow[:,:,1].max() - flow[:,:,1].min())
        # flow[:,:,0] = (flow[:,:,0] - flow[:,:,0].min()) *255 / (flow[:,:,0].max() - flow[:,:,0].min())


        # img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        # warped_mask = warped_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        # # flow = flow[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        # gt = gt[bbox[1]:bbox[3],bbox[0]:bbox[2]]

        # plt.figure()
        # plt.title('display')
        # plt.subplot(311)
        # plt.imshow(img)
        # plt.subplot(312)
        # plt.imshow(warped_mask)
        # plt.subplot(313)
        # plt.imshow(gt)
        # plt.show()

        return data, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])

    def get_img_filename(self, idx):
        print(self.img_list[idx].split('/')[0])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    # transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    dataset = OfflineDataset(train=False, transform=tr.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        img = data['image']
        # print img
        print(img.shape)
        bb
        plt.figure()
        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
        if i == 10:
            break

    plt.show(block=True)