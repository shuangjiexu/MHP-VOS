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
from deeplab.net.generateNet import generate_net
from deeplab.dataloaders import custom_transforms as tr
from deeplab.dataloaders.utils import *

class Deeplabv3plus():
    """
    epoch: 69, 79, 89, 99, 109
    """
    def __init__(self, seqname, obj_id, epoch=99, colorjitter=False,
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
        weight_path = '../prepare/deeplab_model/' + self.seqname + '_' + str(self.obj_id) + '_' + str(self.epoch) + '.pth'
        # weight_path = '/media/shuangjie/ouyang1/VOS/youtube/models/' + self.seqname + '_' + str(self.obj_id) + '_' + str(self.epoch) + '.pth'
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
        mask = np.array(mask, dtype=np.float32)

        img_id = int(img_path.split('/')[-1].split('.')[0])

        flow_dir = os.path.join(self.flow_dir, self.seqname)
        img_dir = os.path.join(self.img_dir, self.seqname)
        # warped_mask, validflowmap01,_,_ = warp_mask(mask, img_id-1, img_id, flow_dir, img_dir)
        warped_mask = (mask > 0.3).astype(np.float32)

        warped_mask = warped_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        warped_mask = cv2.resize(warped_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        img = cv2.imread(img_path)
        img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]

        w_in, h_in = img.shape[:2]
        img = cv2.resize(img, (512,512))
        img = Image.fromarray(img)

        if self.colorjitter:
            img = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(img)

        img = np.subtract(img, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))

        data = np.concatenate((img,np.expand_dims(warped_mask, axis=2)), axis=2)
        data = self.transform(data).unsqueeze(0)

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

            # result = valid_mask(mask, thr=thr)

        return mask

if __name__ == '__main__':
    '''
    obj_id = 1
    deeplab = Deeplabv3plus('carousel', obj_id)
    img_path = '/home/shuangjie/shuangjie/daizong/DAVIS2017/test_dev/JPEGImages/480p/carousel/00059.jpg'
    # mask, _ = load_mask('c', obj_id)
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2])

    bbox = [79, 200, 493, 469] # 23 [453,87,652,273] # compute_bbox_from_mask(mask)

    result = deeplab.compute_mask(img_path, mask, bbox)
    cv2.rectangle(result, pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]), color=(1), thickness=2)
    plt.imshow(result)
    plt.show()
    '''

    obj_id = 2
    deeplab = Deeplabv3plus('carousel', obj_id)
    img_path = '/home/shuangjie/shuangjie/daizong/DAVIS2017/test_dev/JPEGImages/480p/carousel/00000.jpg'
    img = cv2.imread(img_path)
    w_in, h_in = img.shape[:2]
    mask = np.zeros((w_in, h_in))
    bbox = [400,85,620,285]
    bboxes = [[524, 138, 750, 431],[559, 140, 752, 385], [568, 142, 759, 370], [585, 140, 745, 349], 
              [574, 131, 744, 332], [524, 130, 736, 332], [500, 131, 736, 334], [453, 130, 695, 388], 
              [372, 212, 645, 444], [420, 188, 667, 447], [459, 144, 666, 300], [404, 105, 583, 245]]
    # print(w_in,h_in)
    mask, _ = load_mask('/home/shuangjie/shuangjie/daizong/DAVIS2017/test_dev/Annotations/480p/carousel/00000.png', obj_id)
    # print(mask.shape)

    # bbox = compute_bbox_from_mask(mask)

    # result = deeplab.compute_mask(img_path, mask, bboxes[0])
    # img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    # plt.imshow(img)
    # plt.show()

    plt.imshow(mask)
    plt.show()

    for i in range(11):
        i += 1

        img_path = '/home/shuangjie/shuangjie/daizong/DAVIS2017/test_dev/JPEGImages/480p/carousel/'+str(i).zfill(5)+'.jpg'
        mask = deeplab.compute_mask(img_path, mask, bboxes[i])
        bbox = bboxes[i]
        cv2.rectangle(mask, pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]), color=(1), thickness=2)

        plt.imshow(mask)
        plt.show()

