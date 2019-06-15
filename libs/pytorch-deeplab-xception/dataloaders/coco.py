from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import json
from pycocotools.coco import COCO
from coco_transforms import *


class COCODataset(data.Dataset):
    """`COCO Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        year: int number for coco dataset in which year
    """

    def __init__(self, root, train=True, transform=None, year=2014):
        self.root = os.path.expanduser(root)
        self.transform = transform
        # self.target_transform = target_transform
        self.train = train  # training set or test set
        self.year = year

        if self.train:
            self.img_root = os.path.join(self.root, 'train%04d' % year)
            self.json_root = os.path.join(self.root, 'annotations', 'instances_train%04d.json' % year)
        else:
            self.img_root = os.path.join(self.root, 'val%04d' % year)
            self.json_root = os.path.join(self.root, 'annotations', 'instances_val%04d.json' % year)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        with open(self.json_root, 'r') as f:
            self.anno = json.load(f)
        self.info = self.anno['annotations']
        self.coco = COCO(self.json_root)
        self.train_labels = torch.FloatTensor([self.info[index]['category_id'] for index in range(len(self.info))])
        self.test_labels = torch.FloatTensor([self.info[index]['category_id'] for index in range(len(self.info))])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # COCO_train2014_000000233296.jpg
        if self.train:
            img_path = os.path.join(self.img_root, 'COCO_train%04d_%012d.jpg' % (self.year, self.info[index]['image_id']))
        else:
            img_path = os.path.join(self.img_root, 'COCO_val%04d_%012d.jpg' % (self.year, self.info[index]['image_id']))
        target = self.test_labels[index]
        # print(target)

        bbox = self.info[index]['bbox']  # [x1,y1,width,height]

        # solve 0 height sample such as 390267 with skis
        if bbox[2]<2:
            bbox[2] += 2
        if bbox[3]<2:
            bbox[3] += 2

        # solve some gray image
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)
        r, c = img.shape[0:2]
        img = Image.im = Image.fromarray(img[...,:3])
        img = img.crop( [bbox[0], bbox[1], bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1] ) # (x1, y1, x2, y2)
        # img = img.resize((28,28))

        sample = {}
        annIds = self.coco.getAnnIds(self.info[index]['image_id'])
        anns = self.coco.loadAnns(annIds)
        segmentation = np.zeros((r,c),dtype=np.uint8)
        for ann_item in anns:
            mask = self.coco.annToMask(ann_item)
            segmentation[mask>0] = ann_item['category_id']
        if np.max(segmentation)>91:
            print(np.max(segmentation))
            raise ValueError('segmentation > 91')
        sample['segmentation'] = segmentation
        
        sample['image'] = np.array(img)
        # sample['label'] = np.array(target)

        if self.transform is not None:
            sample = self.transform(sample)

        # if self.target_transform is not None:
        #     target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.info)

    def _check_exists(self):
        # print(self.img_root)
        # print(self.json_root)
        return os.path.exists(self.img_root) and os.path.exists(self.json_root)

    def _categories_num(self):
        return len(self.anno['categories'])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    train_data = COCODataset(root='../../cocodata', train=False, transform=ToTensor(), year=2014)
    dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    for ii, sample in enumerate(dataloader):
        img_tmp, segmap = sample['image'], sample['segmentation']
        plt.figure()
        plt.title('display')
        plt.subplot(211)
        plt.imshow(img_tmp)
        plt.subplot(212)
        plt.imshow(segmap)
        break
    plt.show()