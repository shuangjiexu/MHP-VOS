from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize

# from helpers import *
import torch
from torch.utils.data import Dataset
from . import custom_transforms as tr
import torchvision.transforms as transforms

from PIL import Image
from matplotlib import pyplot as plt
from .utils import *

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
#     return flow.astype(np.float32)

# def load_mask(path, obj_id):
#     label = Image.open(path)
#     mask = np.atleast_3d(label)[...,0]
#     mask = mask.copy()
#     mask[mask!=obj_id] = 0
#     mask[mask!=0] = 1
#     return mask.astype(np.float32)#, np.array(label.getpalette())

# def get_obj_num(path):
#     label = Image.open(path)
#     mask = np.atleast_3d(label)[...,0]
#     obj_num = len(np.unique(mask))
#     return obj_num

# cv2.setNumThreads(0)

class OnlineDataset(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='../all_test_file/lucid_mask',
                 flow_root_dir='../all_test_file/lucid_flow',
                 train_val_file = '../DAVIS2017/test_dev/ImageSets/2017',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 obj_id=None,
                 seq_name='blackswan'):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.flow_root_dir = flow_root_dir
        self.transform = transform
        self.meanval = meanval
        self.obj_id = str(obj_id)
        self.seq_name = seq_name

        if self.train:
            fname = 'val'
        else:
            fname = 'val'


        # Initialize the original DAVIS splits for training the parent network
        img_list = []
        labels = []
        flow_list = []
        former_mask = []

        images = np.sort(os.listdir(os.path.join(db_root_dir, seq_name, 'origin')))
        images_path = list(map(lambda x: os.path.join(seq_name, 'origin', x), images))
        flows = np.sort(os.listdir(os.path.join(flow_root_dir, seq_name)))
        flows_path = list(map(lambda x: os.path.join(seq_name, x), flows))

        # obj_num = len(os.listdir(os.path.join(db_root_dir, seq_name))) - 1
        # for obj_index in range(obj_num):
        # img_list.extend(images_path)
        # flow_list.extend(flows_path)

        lab_path = []
        lab_f_path = []
        for num200 in range(200):
            lab = os.path.join(seq_name, self.obj_id, str(num200).zfill(5)+'_gt2.png')
            check_mask = get_obj_num(os.path.join(self.db_root_dir, seq_name.strip(), self.obj_id, str(num200).zfill(5)+'_gt2.png'))
            if check_mask == 1:
                continue
            area_mask, _ = imread_indexed(os.path.join(self.db_root_dir, seq_name.strip(), self.obj_id, str(num200).zfill(5)+'_gt2.png'))
            mask_area = valid_area(area_mask)
            if mask_area < 20*20:
                continue
            # if get_obj_num(os.path.join(self.db_root_dir, lab)) == 1:
            #    continue
            img_list.append(images_path[num200])
            # flow_list.append(flows_path[num200])
            lab_path.append(lab)
            lab_f = os.path.join(seq_name, self.obj_id, str(num200).zfill(5)+'_gt1.png')
            check_mask = get_obj_num(os.path.join(self.db_root_dir, seq_name.strip(), self.obj_id, str(num200).zfill(5)+'_gt1.png'))
            if check_mask == 1:
                lab_f = lab
            else:
                pass
            lab_f_path.append(lab_f)
        labels.extend(lab_path)
        former_mask.extend(lab_f_path)
        if len(labels) % 2 != 0:
            img_list.append(img_list[-1])
            # flow_list.append(flows_path[num200])
            labels.append(labels[-1])
            former_mask.append(former_mask[-1])
        #print(len(labels))
        # img_list = [os.path.join('../DAVIS2017/trainval/JPEGImages/480p',seq_name,'00000.jpg'),os.path.join('../DAVIS2017/trainval/JPEGImages/480p',seq_name,'00000.jpg')]
        # labels = [os.path.join('../DAVIS2017/trainval/Annotations/480p',seq_name,'00000.png'),os.path.join('../DAVIS2017/trainval/Annotations/480p',seq_name,'00000.png')]
        # flow_list = [os.path.join('../trainval_flow',seq_name, '00001.flo'),os.path.join('../trainval_flow',seq_name, '00001.flo')]
        # former_mask = [os.path.join('../DAVIS2017/trainval/Annotations/480p',seq_name,'00000.png'),os.path.join('../DAVIS2017/trainval/Annotations/480p',seq_name,'00000.png')]

        
        # print len(labels)
        # print len(img_list)
        # print len(flow_list)
        # print len(former_mask)
        # bb

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        # print(labels[0])
        # print(img_list[0])
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

        # print(sample['image'].shape)
        # print(sample['mask'].shape)
        # print(sample['flow'].shape)
        # print(sample['label'].shape)
        # bb
        # sample['image'] = torch.cat([sample['image'], sample['mask']], 0)
        # sample.pop('mask')
        # sample.pop('flow')

        # if self.train:
        #     sample['image'] = sample['image']
        #     sample['label'] = sample['label']
            # sample['image'] = torch.cat([sample['image'], sample['image']], 0)
            # sample['label'] = torch.cat([sample['label'], sample['label']], 0)
        # print(sample['image'].shape)
        # bb
        # self.get_img_filename(idx)
        # print(sample['image'].shape, sample['label'].shape)
        return sample
        # img, gt = self.make_img_gt_pair(idx)

        # sample = {'image': img, 'gt': gt}

        # # if self.seq_name is not None:
        # #     fname = os.path.join(self.seq_name, "%05d" % idx)
        # #     sample['fname'] = fname

        # if self.transform is not None:
        #     sample = self.transform(sample)

        # return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        label_ ,_ = imread_indexed(os.path.join(self.db_root_dir, self.labels[idx]))
        label = label_.copy()
        gt = np.array(label, dtype=np.float32)

        bbox = compute_bbox_from_mask(gt)
        gt = gt[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        #print(bbox)
        #print(gt.shape)
        #print(self.labels[idx])
        gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)

        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        # img = Image.open(os.path.join(self.db_root_dir, self.img_list[idx]))
        img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        img = cv2.resize(img, (512,512))
        img = Image.fromarray(img)
        img = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(img)
        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        # print(self.db_root_dir)
        

        f_mask_ ,_ = imread_indexed(os.path.join(self.db_root_dir, self.former_mask[idx]))
        f_mask = f_mask_.copy()
        mask = np.array(f_mask, dtype=np.float32)

        im2_id = int(self.img_list[idx].split('/')[-1].split('.')[0])
        im1_id = im2_id
        obj_id = int(self.labels[idx].split('/')[1])

        flow_dir = os.path.join(self.flow_root_dir, self.seq_name)
        img_dir = os.path.join('../all_test_file/test_lucid_dataset', self.seq_name)
        warped_mask, validflowmap01,_,_ = warp_mask_lucid(mask, im1_id, im2_id, flow_dir, img_dir)
        warped_mask = (warped_mask > 0.3).astype(np.float32)
        warped_mask = warped_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        warped_mask = cv2.resize(warped_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        data = np.concatenate((img,np.expand_dims(warped_mask, axis=2)), axis=2)
        
        # mask = mask*255
        # mask = np.expand_dims(mask, axis=2)

        # flow = read_flow(os.path.join(self.flow_root_dir, self.flow_list[0]))
        # flow[:,:,1] = (flow[:,:,1] - flow[:,:,1].min()) *255 / (flow[:,:,1].max() - flow[:,:,1].min())
        # flow[:,:,0] = (flow[:,:,0] - flow[:,:,0].min()) *255 / (flow[:,:,0].max() - flow[:,:,0].min())

        # img = np.concatenate((img, mask, flow), axis=2)
        # print img.shape
        # print gt.shape
        # print mask.shape
        # print flow.shape
        # bb

        # plt.figure()
        # plt.title('display')
        # plt.subplot(311)
        # plt.imshow(img)
        # plt.subplot(312)
        # plt.imshow(warped_mask)
        # plt.subplot(313)
        # plt.imshow(gt)
        # plt.show()
        # bb

        # print(img.shape)
        # print(mask.shape)
        # print(flow.shape)
        # print(gt.shape)

        return data, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    dataset = OnlineDataset(train=True, transform=tr.ToTensor())
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
