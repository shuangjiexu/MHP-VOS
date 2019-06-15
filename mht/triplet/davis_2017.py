import os
import numpy as np
import random
from collections import OrderedDict

import torch.utils.data
import torchvision.transforms as transforms
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
from dataset.syntheseis import synthesis

def get_obj_num(path):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    obj_num = len(np.unique(mask))
    return obj_num
def load_mask(path, obj_id):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    mask = mask.copy()
    mask[mask!=obj_id] = 0
    mask[mask!=0] = 1
    return mask.astype(np.float32), np.array(label.getpalette())
def valid_area(mask):
    return np.sum((mask>0).astype(np.int))

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, transform=None, train=True):
        """
        base_path: base path of davis-2017
        return three images (img1: anchor, img2: far, img3: close)
        For three image 'img1 img2 img3', a triplet is defined such that image img1 is more
        similar to image img3 than it is to image img2, e.g.,
        """
        self.base_path = base_path
        self.info = self.read_all_info()
        train_num = int(len(self.info.keys())*0.8)
        train_class = self.info.keys()[:train_num]
        test_class = self.info.keys()[train_num:]
        if train:
            self.filenamelist = self.read_info_list(self.info, train_class)
            random.shuffle(self.filenamelist)
        else:
            self.filenamelist = self.read_info_list(self.info, test_class)
        self.transform = transform

    def read_all_info(self):
        indexed_path = os.path.join(self.base_path)#, 'IndexedSegmentation', '480p')
        # read all obj frames into one dict
        info = OrderedDict()
        for class_name in os.listdir(indexed_path):
            class_info = {}
            class_path = os.path.join(indexed_path, class_name)
            for obj_name in os.listdir(class_path):
                if obj_name == 'origin':
                    continue
                save_list = []
                for frame_seq in os.listdir(os.path.join(class_path, obj_name)):
                    check_mask = get_obj_num(os.path.join(self.base_path, class_name, obj_name, frame_seq))
                    if check_mask == 1:
                        continue
                    loadmask,_ = load_mask(os.path.join(self.base_path, class_name, obj_name, frame_seq), 1)
                    mask_area = valid_area(loadmask)
                    if mask_area < 20*20:
                        continue
                    # rows = np.any(loadmask, axis=1)
                    # cols = np.any(loadmask, axis=0)
                    # rmin, rmax = np.where(rows)[0][[0, -1]]
                    # cmin, cmax = np.where(cols)[0][[0, -1]]
                    # if cmin == cmax or rmin == rmax:
                    #     print(class_name, obj_name, frame_seq)
                    #     print(cmin, cmax, rmin, rmax)
                    #     print(self.compute_bbox(loadmask))
                    save_list.append(frame_seq)
                class_info[obj_name] = (save_list)
            info[class_name] = (class_info)
        return info

    def read_info_list(self, info, class_list):
        # get list for training
        info_list = []
        for class_name in class_list:
            for obj_name in info[class_name]:
                for seq in info[class_name][obj_name]:
                    info_list.append( {'class': class_name, 'obj': obj_name, 'frame': seq} )
        return info_list

    def compute_bbox(self, mask, random_pad=0.1):
        # mask: h*w with value of 0,1
        # return in (y1,y2,x1,x2)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # some data is a line (such as lindy-hop 70 frame. add 5 piexl in each side)
        add_pixel = 5
        cmin = max(0, cmin-add_pixel)
        rmin = max(0, rmin-add_pixel)
        cmax = min(cmax+add_pixel, mask.shape[1]-1)
        rmax = min(rmax+add_pixel, mask.shape[0]-1)

        if random_pad:
            r_d = int((rmax - rmin) * random_pad)
            c_d = int((cmax - cmin) * random_pad)
            rmin = max(0, rmin-random.randint(0,r_d))                    # y1
            cmin = max(0, cmin-random.randint(0,c_d))                    # x1
            rmax = min(mask.shape[0]-1, rmax+random.randint(0,r_d))      # y2
            cmax = min(mask.shape[1]-1, cmax+random.randint(0,c_d))      # x2

        return rmin, rmax, cmin, cmax

    def synthesis(self, img, mask):
        # change input RGB skimage to BGR cv2 image
        im = img[:,:,::-1]
        img_aug, mask_aug = synthesis(im, mask)
        return img_aug[:,:,::-1], mask_aug

    def load_anchor(self, img, mask):
        bbox = self.compute_bbox(mask)
        return img[bbox[0]:bbox[1], bbox[2]:bbox[3],:]

    def load_far(self, img, mask, random_pad=0.1):
        # crop from the other position in img
        bbox = self.compute_bbox(mask)
        w_d = bbox[3]-bbox[2] + random.randint(0, int((bbox[3]-bbox[2]) * random_pad))
        h_d = bbox[1]-bbox[0] + random.randint(0, int((bbox[1]-bbox[0]) * random_pad))

        x = int(np.random.uniform(0, img.shape[1]-1))
        y = int(np.random.uniform(0, img.shape[0]-1))

        tmp_bbox = (max(0,x-w_d/2), min(img.shape[1]-1, x+w_d/2), max(0, y-h_d/2), min(img.shape[0]-1, y+h_d/2))
        iou = get_iou({'x1':bbox[2], 'x2':bbox[3], 'y1':bbox[0], 'y2':bbox[1]},
                {'x1':tmp_bbox[0], 'x2':tmp_bbox[1], 'y1':tmp_bbox[2], 'y2':tmp_bbox[3]})
        if iou > 0.5:
            return self.load_far(img, mask)
        else:
            return img[tmp_bbox[2]:tmp_bbox[3], tmp_bbox[0]:tmp_bbox[1],:]

    def load_far_other(self, filename):
        obj_number = len(self.info[filename['class']].keys())
        if obj_number > 1:
            obj_select = self.info[filename['class']].keys()
            obj_select.remove(filename['obj'])
            obj_id = random.choice(obj_select)
            file = filename['class']

            total = 0
            while len(self.info[file][obj_id][:]) == 0:
                obj_id = random.choice(self.info[file].keys())
                total += 1
                if total>20:
                    class_index = self.info.keys().index(filename['class'])
                    file = random.choice(self.info.keys())
                    while file == filename['class']:
                        file = random.choice(self.info.keys())

                    obj_id = random.choice(self.info[file].keys())
                    while obj_id == 'origin':
                        obj_id = random.choice(self.info[file].keys())
                    while len(self.info[file][obj_id][:]) == 0:
                        obj_id = random.choice(self.info[file].keys())
                    break
        else:
            class_index = self.info.keys().index(filename['class'])
            file = random.choice(self.info.keys())
            while file == filename['class']:
                file = random.choice(self.info.keys())

            obj_id = random.choice(self.info[file].keys())
            while obj_id == 'origin':
                obj_id = random.choice(self.info[file].keys())
            while len(self.info[file][obj_id][:]) == 0:
                obj_id = random.choice(self.info[file].keys())

        frame_list = self.info[file][obj_id][:]
        # print(file, obj_id)
        # print(frame_list)
        frame = random.choice(frame_list)
        # print(frame)
        img_other = io.imread(os.path.join(
            self.base_path, file, 'origin', frame[:-3]+'jpg'))
        mask_other = np.array(Image.open(os.path.join(
            self.base_path, file, obj_id, frame)))
        bbox = self.compute_bbox(mask_other)
        return img_other[bbox[0]:bbox[1], bbox[2]:bbox[3],:]


    def load_close(self, filename):
        # object in other frame
        frame_list = self.info[filename['class']][filename['obj']][:]
        frame_list.remove(filename['frame'])
        if len(frame_list) == 0:
            frame = filename['frame']
        else:
            frame = random.choice(frame_list)
        img_other = io.imread(os.path.join(
            self.base_path, filename['class'], 'origin', frame[:-3]+'jpg'))
        mask_other = np.array(Image.open(os.path.join(
            self.base_path, filename['class'], filename['obj'], frame)))
        bbox = self.compute_bbox(mask_other)
        return img_other[bbox[0]:bbox[1], bbox[2]:bbox[3],:]

    def __getitem__(self, index):
        filename = self.filenamelist[index]
        img_orig = io.imread(os.path.join(
            self.base_path, filename['class'], 'origin', filename['frame'][:-3]+'jpg'))
        mask = np.array(Image.open(os.path.join(
            self.base_path, filename['class'], filename['obj'], filename['frame'])))
        use_trans_pos = random.randint(0,1)
        use_trans_neg = random.randint(0,1)
        img_aug, mask_aug = self.synthesis(img_orig, mask)

        # img1: anchor, img2: far, img3: close
        img1 = self.load_anchor(img_orig, mask)
        # plt.imshow(img1)
        # plt.show()
        # img2 = self.load_far(img_aug, mask_aug) if use_trans_neg else self.load_far(img_orig, mask)
        img2 = self.load_far_other(filename) if use_trans_neg else self.load_far(img_orig, mask)
        # plt.imshow(img2)
        # plt.show()
        img3 = self.load_anchor(img_aug, mask_aug) if use_trans_pos else self.load_close(filename)
        # plt.imshow(img3)
        # plt.show()
        # change from skimage to PIL image
        img_1 = Image.fromarray(img1, 'RGB')
        img_2 = Image.fromarray(img2, 'RGB')
        img_3 = Image.fromarray(img3, 'RGB')

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)

        return img_1, img_2, img_3

    def __len__(self):
        return len(self.filenamelist)

# -----------------------------util------------------------------
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == '__main__':
    # test code: save data to check
    davis_2017 = TripletImageLoader('../video_segmentation/multi_mask')
    davis_2017.__getitem__(1)
    for i in range(10):
        print('sloving: '+str(i)+'-------------')
        sample = davis_2017[i]
        print(sample[0].size)
        print(sample[1].size)
        print(sample[2].size)
        sample[0].save(str(i)+'_1.jpg')
        sample[1].save(str(i)+'_2.jpg')
        sample[2].save(str(i)+'_3.jpg')
