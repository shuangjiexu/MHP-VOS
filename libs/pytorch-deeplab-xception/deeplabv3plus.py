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
from net.generateNet import generate_net
from dataloaders import custom_transforms as tr
from dataloaders.utils import *

class Deeplabv3plus():
    """
    epoch: 69, 79, 89, 99, 109
    """
    def __init__(self, seqname, obj_id, epoch=99, colorjitter=True,
                    flow_dir='../data/all_test_file/test_flow', 
                        img_dir='../data/DAVIS/test_dev/JPEGImages/480p'):
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
        weight_path = './model/' + self.seqname + '_' + str(self.obj_id) + '_' + str(self.epoch) + '.pth'
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
        warped_mask, validflowmap01,_,_ = warp_mask(mask, img_id-1, img_id, flow_dir, img_dir)
        warped_mask = (warped_mask > 0.3).astype(np.float32)

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

            result = valid_mask(mask, thr=thr)

        return result

if __name__ == '__main__':
    obj_id = 2
    deeplab = Deeplabv3plus('carousel', obj_id)
    img_path = '../data/DAVIS/test_dev/JPEGImages/480p/carousel/00001.jpg'
    img = cv2.imread(img_path)
    w_in, h_in = img.shape[:2]
    mask = np.zeros((w_in, h_in))
    bbox = [400,85,620,285]
    bboxes = [[524,138,750,431],[540,130,765,415],[547,132,772,398],[547,130,772,379],
              [555,126,763,355],[533,121,755,340],[526,118,734,324],[505,115,703,309],
              [481,116,691,288],[447,111,669,273],[423,107,648,262],[388,102,596,252]]
    # print(w_in,h_in)
    mask, _ = load_mask('../data/DAVIS/test_dev/Annotations/480p/carousel/00000.png', obj_id)
    # print(mask.shape)

    # bbox = compute_bbox_from_mask(mask)

    result = deeplab.compute_mask(img_path, mask, bboxes[0])
    # img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
    # plt.imshow(img)
    # plt.show()

    plt.imshow(result)
    plt.show()

    for i in range(11):
        i += 1

        mask = result
        img_path = '../data/DAVIS/test_dev/JPEGImages/480p/carousel/'+str(i).zfill(5)+'.jpg'
        result = deeplab.compute_mask(img_path, mask, bboxes[i])

        plt.imshow(result)
        plt.show()

