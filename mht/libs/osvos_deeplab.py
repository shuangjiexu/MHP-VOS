import os
import cv2
import numpy as np
from PIL import Image

# PyTorch includes
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# Custom includes
from osvos.net.generateNet import generate_net
from osvos.dataloaders import custom_transforms as tr
from osvos.dataloaders.utils import *

class DeeplabOSVOS():
    """
    epoch: 69, 79, 89, 99, 109
    """
    def __init__(self, seqname, obj_id, epoch=99, colorjitter=True,
                    flow_dir='../prepare/test_flow', 
                        img_dir='../prepare/DAVIS_2017/JPEGImages/480p'):
        torch.cuda.manual_seed(1701)

        self.seqname = seqname
        self.obj_id = obj_id
        self.epoch = epoch
        self.flow_dir = flow_dir
        self.img_dir = img_dir
        self.colorjitter = colorjitter

        self.net = generate_net()
        self.net = nn.DataParallel(self.net).to(torch.device('cuda'))
        self.net.load_state_dict(self.load_weight())
        self.net.eval()

    def load_weight(self):
        weight_path = 'osvos/model/' + self.seqname + '_' + str(self.obj_id) + '_' + str(self.epoch) + '.pth'
        return torch.load(weight_path)

    def transform(self, tmp):
        if tmp.ndim == 2:
            tmp = tmp[:, :, np.newaxis]

        tmp = tmp.transpose((2, 0, 1))
        return torch.from_numpy(tmp)

    def get_data(self, img_path, mask, bbox):
        """
        mask: mask of Previous frame
        """
        img_id = int(img_path.split('/')[-1].split('.')[0])

        img = cv2.imread(img_path)
        img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]

        w_in, h_in = img.shape[:2]
        img = cv2.resize(img, (512,512))
        img = Image.fromarray(img)

        if self.colorjitter:
            img = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(img)

        img = np.subtract(img, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))

        data = self.transform(img).unsqueeze(0)

        return data, (h_in, w_in)

    def compute_mask(self, img_path, mask, bbox, thr=0.3):
        """
        mask: mask of Previous frame
        """
        h, w = mask.shape
        data, crop_size = self.get_data(img_path, mask, bbox)

        with torch.no_grad():
            data = Variable(data.cuda())
            outputs = self.net.forward(data)

            pred = np.transpose(outputs.cpu().data.numpy()[0, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)

            mask_local = cv2.resize(pred, crop_size, interpolation=cv2.INTER_NEAREST)
            mask = np.zeros((h,w)).astype(np.float32)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask_local

            result = valid_mask(mask, thr=thr)

        return result

if __name__ == '__main__':
    obj_id = 2
    deeplab = DeeplabOSVOS('carousel', obj_id)
    img_path = '../prepare/DAVIS_2017/JPEGImages/480p/carousel/00008.jpg'
    img = cv2.imread(img_path)
    w_in, h_in = img.shape[:2]
    mask = np.zeros((w_in, h_in))
    bbox = [472,110,696,299]
    # print(w_in,h_in)
    # mask, _ = load_mask('../DAVIS2017/test_dev/Annotations/480p/aerobatics/00000.png', obj_id)
    # print(mask.shape)

    # bbox = compute_bbox_from_mask(mask)

    result = deeplab.compute_mask(img_path, mask, bbox)
    img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    plt.imshow(img)
    plt.show()

    plt.imshow(result)
    plt.show()

