import os
import cv2
from PIL import Image

import skimage.measure as measure

import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=1):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou

def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss
    
def get_obj_num(path):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    obj_num = len(np.unique(mask))
    return obj_num

def imread_indexed(filename):
    """ Load image given filename."""

    im = Image.open(filename)

    annotation = np.atleast_3d(im)[...,0]
    return annotation,np.array(im.getpalette()).reshape((-1,3))

def imwrite_index(filename,array,color_palette):
    """ Save indexed png."""

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array*255)
    im =im.convert("P")
    # im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

def valid_area(mask):
    return np.sum((mask>0).astype(np.int))

def valid_mask(mask):
    # write origion mask
    mask_valid = mask.copy()
    # print(mask_valid.max())
    mask_valid[mask_valid>0.3] = 1
    mask_valid[mask_valid<=0.3] = 0
    return mask_valid.astype(np.float32)

def load_mask(path, obj_id):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    mask = mask.copy()
    mask[mask!=obj_id] = 0
    mask[mask!=0] = 1
    return mask.astype(np.float32), np.array(label.getpalette())

def transform_i(tmp):
    if tmp.ndim == 2:
        flagval = cv2.INTER_NEAREST
    else:
        flagval = cv2.INTER_CUBIC
    # print(tmp.shape)
    # sc = 512
    # tmp = cv2.resize(tmp, (sc,sc), interpolation=flagval)

    if tmp.ndim == 2:
        tmp = tmp[:, :, np.newaxis]

        # swap color axis because
        # numpy image: H x W x C
         # torch image: C X H X W

    tmp = tmp.transpose((2, 0, 1))
    return torch.from_numpy(tmp)

def read_flow(flowfile):
    f = open(flowfile, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)

def make_img_gt_pair(img_path, mask, seq_name, label_path, obj_id, transform=transform_i):
        """
        Make the image-ground-truth pair
        """
        mask = np.array(mask, dtype=np.float32)

        im2_id = int(img_path.split('/')[-1].split('.')[0])
        im1_id = im2_id - 1
        obj_id = obj_id

        flow_dir = os.path.join('../all_test_file/test_flow', seq_name)
        img_dir = os.path.join('../DAVIS2017/test_dev/JPEGImages/480p', seq_name)
        warped_mask, validflowmap01,_,_ = warp_mask(mask, im1_id, im2_id, flow_dir, img_dir)
        warped_mask = (warped_mask > 0.3).astype(np.float32)

        bbox = compute_bbox_from_mask(warped_mask)

        warped_mask = warped_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        warped_mask = cv2.resize(warped_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        img = cv2.imread(img_path)

        img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]

        w_in, h_in = img.shape[:2]
        img = cv2.resize(img, (512,512))
        img = Image.fromarray(img)
        # img = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(img)
        # img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))

        data = np.concatenate((img,np.expand_dims(warped_mask, axis=2)), axis=2)
   
        label_ ,palette = load_mask(label_path, obj_id)
        label = label_.copy()
        gt = np.array(label, dtype=np.float32)
        gt = gt[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        gt = cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST)
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


        # flow = read_flow(os.path.join(flow_path))
        # flow[:,:,1] = (flow[:,:,1] - flow[:,:,1].min()) *255 / (flow[:,:,1].max() - flow[:,:,1].min())
        # flow[:,:,0] = (flow[:,:,0] - flow[:,:,0].min()) *255 / (flow[:,:,0].max() - flow[:,:,0].min())

        # flow = flow[bbox[1]:bbox[3],bbox[0]:bbox[2],:]

        data = transform(data)
        gt = transform(gt)
        # print(flow.shape)
        data = data.unsqueeze(0)
        gt = gt.unsqueeze(0)
       

        return data, gt, bbox, (h_in,w_in), palette

def clip_around_bbox(bbox_in, imgsize):
    """ 
    bbox is in [[x1, y1], [x2, y2]]
    imgsize is in [h, w]
    """
    bbox_around = np.around(bbox_in)
    bbox = np.zeros((4))
    bbox[0] = max(0, bbox_around[0])
    bbox[1] = max(0, bbox_around[1])
    bbox[2] = min(imgsize[1]-1, bbox_around[2])
    bbox[3] = min(imgsize[0]-1, bbox_around[3])
    return bbox.astype(np.int16)

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width]. Mask pixels are either 1 or 0.

    Returns: bbox array [x1, y1, x2, y2].
    """
    # Bounding box.
    regions = measure.regionprops(mask)
    if len(regions):
        y1,x1,y2,x2 = regions[0].bbox 
        bbox = np.array([x1, y1, x2, y2])
        return bbox.astype(np.int32)
    else:
        return np.array([0,0,0,0])

def compute_bbox_from_mask(mask):
    # return [x1, y1, x2, y2]
    margin = 0.15
    h,w = mask.shape[:2]
    # [x1, y1, x2, y2]
    bbox = extract_bboxes((mask>0.1).astype(np.int16))
    h_box = bbox[3] - bbox[1]
    w_box = bbox[2] - bbox[0]
    # apply margin
    bbox_margined = [bbox[0]-margin*w_box, bbox[1]-margin*h_box, bbox[2]+margin*w_box, bbox[3]+margin*h_box]
    bbox_margined = clip_around_bbox(bbox_margined, mask.shape[:2])
    return bbox_margined

def warp_back(img, flow, residueimg, validflowmap01):
    h, w = flow.shape[:2]
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_CUBIC, borderMode =cv2.BORDER_CONSTANT )
    validflowmap01[np.isnan(res)] = 0 # interp might cause NaN when indexing out of range
    validflowmap01[residueimg > 0.3] = 0 # add additional validmap using residueimg

    # ONLY reset nan pixels, do not reset invalid pixels
    res[np.isnan(res)] = 0
    return res, validflowmap01

def checkflow(flow01, flow10, th=2):
    h,w,c = flow01.shape
    x,y = np.meshgrid(np.arange(w),np.arange(h))
    xid01 = np.clip((x+flow01[:,:,0]).astype(np.int16),0,w-1)
    yid01 = np.clip((y+flow01[:,:,1]).astype(np.int16),0,h-1)
    xid10 = np.clip((x+flow10[:,:,0]).astype(np.int16),0,w-1)
    yid10 = np.clip((y+flow10[:,:,1]).astype(np.int16),0,h-1)
    outofmap01 = ((x+flow01[:,:,0]).astype(np.int16) < 0) | ((y+flow01[:,:,1]).astype(np.int16) <0) | ((x+flow01[:,:,0]).astype(np.int16) >= w) | ((y+flow01[:,:,1]).astype(np.int16) >= h)
    outofmap10 = ((x+flow10[:,:,0]).astype(np.int16) < 0) | ((y+flow10[:,:,1]).astype(np.int16) <0) | ((x+flow10[:,:,0]).astype(np.int16) >= w) | ((y+flow10[:,:,1]).astype(np.int16) >= h)

    flow01_u = flow01[:,:,0]
    flow01_v = flow01[:,:,1]
    flow10_u = flow10[:,:,0]
    flow10_v = flow10[:,:,1]

    idx01_outlier_x = abs(flow10_u[yid01,xid01]+flow01[:,:,0]) > th
    idx01_outlier_y = abs(flow10_v[yid01,xid01]+flow01[:,:,1]) > th
    idx01_outlier = idx01_outlier_x | idx01_outlier_y

    idx10_outlier_x = abs(flow01_u[yid10,xid10]+flow10[:,:,0]) > th
    idx10_outlier_y = abs(flow01_v[yid10,xid10]+flow10[:,:,1]) > th
    idx10_outlier = idx10_outlier_x | idx10_outlier_y

    validflowmap01 = np.ones((h,w))
    validflowmap10 = np.ones((h,w))

    validflowmap01[(idx01_outlier!=0) | (outofmap01!=0)] = 0
    validflowmap10[(idx10_outlier!=0) | (outofmap10!=0)] = 0
    return validflowmap01, validflowmap10

def warp_mask(mask, im1_id, im2_id, flow_dir, img_dir):
    # image load
    img1 = cv2.imread(os.path.join(img_dir, '%05d.jpg'%im1_id))
    img2 = cv2.imread(os.path.join(img_dir, '%05d.jpg'%im2_id))
    # flow and warp load
    flow_01_file = os.path.join(flow_dir, 'flownet2_%05d_%05d.flo'%(im1_id, im2_id))  # flownet2_00070_00069.flo
    flow_10_file = os.path.join(flow_dir, 'flownet2_%05d_%05d.flo'%(im2_id, im1_id)) 
    warp_01_file = os.path.join(flow_dir, 'flownet2_%05d_%05d.png'%(im1_id, im2_id))  # flownet2_00070_00069.png
    warp_10_file = os.path.join(flow_dir, 'flownet2_%05d_%05d.png'%(im2_id, im1_id))

    flow01 = read_flow(flow_01_file)
    flow10 = read_flow(flow_10_file)
    warpI01 = cv2.imread(warp_01_file).astype(np.float32)
    warpI10 = cv2.imread(warp_10_file).astype(np.float32)

    residueimg21 = np.max(abs(warpI10 - img2), axis=2)/255.0 # maximum residue from rgb channels
    validflowmap01, validflowmap10 = checkflow(flow01, flow10)
    warped_mask, validflowmap10 = warp_back(mask.astype(np.float32), flow10, residueimg21, validflowmap10)
    return warped_mask, validflowmap10, flow01, validflowmap01

def warp_mask_lucid(mask, im1_id, im2_id, flow_dir, img_dir):
    # image load
    img1 = cv2.imread(os.path.join(img_dir, '%05d_rgb1.jpg'%im1_id))
    img2 = cv2.imread(os.path.join(img_dir, '%05d_rgb2.jpg'%im2_id))
    # flow and warp load
    flow_01_file = os.path.join(flow_dir, 'flownet2_%05d_%05d.flo'%(im1_id, im2_id+1))  # flownet2_00070_00069.flo
    flow_10_file = os.path.join(flow_dir, 'flownet2_%05d_%05d.flo'%(im2_id+1, im1_id)) 
    warp_01_file = os.path.join(flow_dir, 'flownet2_%05d_%05d.png'%(im1_id, im2_id+1))  # flownet2_00070_00069.png
    warp_10_file = os.path.join(flow_dir, 'flownet2_%05d_%05d.png'%(im2_id+1, im1_id))

    flow01 = read_flow(flow_01_file)
    flow10 = read_flow(flow_10_file)
    warpI01 = cv2.imread(warp_01_file).astype(np.float32)
    warpI10 = cv2.imread(warp_10_file).astype(np.float32)

    residueimg21 = np.max(abs(warpI10 - img2), axis=2)/255.0 # maximum residue from rgb channels
    validflowmap01, validflowmap10 = checkflow(flow01, flow10)
    warped_mask, validflowmap10 = warp_back(mask.astype(np.float32), flow10, residueimg21, validflowmap10)
    return warped_mask, validflowmap10, flow01, validflowmap01

if __name__ == '__main__':
    dataroot = '../../multi_mask/'
    filelist = os.listdir(dataroot)
    for file in filelist:
        print(file)
        dataroot1 = dataroot+file
        filelist1 = os.listdir(dataroot1)
        for num in range(len(filelist1)-1):
            dataroot2 = dataroot1+'/'+str(num+1)
            filelist2 = os.listdir(dataroot2)
            for maskfile in filelist2:
                maskroot = dataroot2+'/'+maskfile
                mask,_ = imread_indexed(maskroot)
                mask = np.array(mask, dtype=np.float32)
                bbox = compute_bbox_from_mask(mask)
                if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                    if bbox[0] == bbox[1] == bbox[2] == bbox[3] == 0:
                        print(bbox)
                        continue
                    else:
                        print(maskroot)
                        print(bbox)