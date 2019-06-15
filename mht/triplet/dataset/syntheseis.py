import os
import numpy as np

# require install skinpaint in https://github.com/soupault/scikit-inpaint
# by pip install git+https://github.com/soupault/scikit-inpaint.git --user
# import skinpaint
from skimage import io
from skimage.color import rgb2gray
from PIL import Image
# from matting.poisson_matting import poisson_matte
import cv2
import random
import matplotlib.pyplot as plt
# get torchvision from source for RandomAffine:  
# git clone https://github.com/pytorch/vision.git, cd vision, python setup.py install
import torchvision.transforms as transforms

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def get_transform(img_shape, is_tps=False):
    '''
    get cv2 image numpy ndarray
    '''
    # apply the thin-plate splines for mask
    rows=img_shape[0]
    cols=img_shape[1]
    tps = None
    if is_tps:
        tps = get_tps(img_shape)

    # return transform
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad((int(rows * 0.15), int(cols * 0.15), int(rows * 0.15), int(cols * 0.15))),
        transforms.CenterCrop((rows, cols))
        # transforms.RandomAffine(30, translate=(0.1,0.1), scale=(0.85,1.15), shear=10)
    ])
    # Rotation
    random_rotation = random.randint(-30,30)
    M_rotation = cv2.getRotationMatrix2D((cols/2,rows/2),random_rotation,1)
    # Translation
    w_trans = random.randint(-int(0.1*cols),int(0.1*cols))
    h_trans = random.randint(-int(0.1*rows),int(0.1*rows))
    M_translation = np.float32([[1,0,w_trans],[0,1,h_trans]])
    # Scale
    scale_h = int(rows * random.uniform(0.85, 1.15))
    scale_w = int(cols * random.uniform(0.85, 1.15))

    def aug_op(img):
        dst_rotation = cv2.warpAffine(img,M_rotation,(cols,rows))
        dst_translation = cv2.warpAffine(dst_rotation,M_translation,(cols,rows))
        res = cv2.resize(dst_translation,(scale_w, scale_h), interpolation = cv2.INTER_CUBIC)
        # pad and crop
        out_img = PIL2array(transform(res.astype(np.uint8)))
        if tps:
            out_img = tps.warpImage(img)
        return out_img
    return aug_op

def get_tps(img_shape, ratio = 0.1):
    tps = cv2.createThinPlateSplineShapeTransformer()
    # get bbox, the thin-plate splines applied by 10% of bbox size
    w = img_shape[1]
    h = img_shape[0]
    x = img_shape[1]
    y = img_shape[0]
    trans_w = int(w*ratio)
    trans_h = int(h*ratio)
    sshape = np.array([[trans_w,trans_h],
                        [w-1-trans_w,trans_h],
                        [trans_w,h-1-trans_h],
                        [w-1-trans_w,h-1-trans_h]],np.float32)
    tshape = np.array([[trans_w+random.randint(-trans_w,trans_w),trans_h+random.randint(-trans_h,trans_h)],
                        [w-1-trans_w+random.randint(-trans_w,trans_w),trans_h+random.randint(-trans_h,trans_h)],
                        [trans_w+random.randint(-trans_w,trans_w),h-1-trans_h+random.randint(-trans_h,trans_h)],
                        [w-1-trans_w+random.randint(-trans_w,trans_w),h-1-trans_h+random.randint(-trans_h,trans_h)]],np.float32)
    sshape = sshape.reshape(1,-1,2)
    tshape = tshape.reshape(1,-1,2)

    matches = list()
    matches.append(cv2.DMatch(0,0,0))
    matches.append(cv2.DMatch(1,1,0))
    matches.append(cv2.DMatch(2,2,0))
    matches.append(cv2.DMatch(3,3,0))

    tps.estimateTransformation(sshape,tshape,matches)
    return tps

def get_padding(in_shape, out_shape):
    pad_h = out_shape[0] - in_shape[0]
    pad_w = out_shape[1] - in_shape[1]
    left_pad = random.randint(0,pad_w)
    top_pad = random.randint(0,pad_h)
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad((left_pad, top_pad, pad_w-left_pad, pad_h-top_pad))
    ])
    def random_padding(img):
        return PIL2array(transform(img.astype(np.uint8)))
    return random_padding

def compute_bbox(mask, random_pad=0.1):
    # mask: h*w with value of 0,1
    # return in (y1,y2,x1,x2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    # plt.imshow(mask)
    # plt.show()
    # print(np.where(rows))
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

def synthesis(img, mask):
    '''
    img and mask are read by cv2 (BGR)
    '''
    # remove mask and inpaint img to get bg
    mask = mask.astype(np.uint8)
    image_defect = img.copy()
    for layer in range(image_defect.shape[-1]):
        image_defect[np.where(mask)] = 0
    # img_inpainted = skinpaint.criminisi(image_defect, mask, multichannel=True)
    img_inpainted = cv2.inpaint(image_defect,mask,3,cv2.INPAINT_TELEA)
    # some transform to bg 
    tranform_bg = get_transform(img_inpainted.shape)
    bg = tranform_bg(img_inpainted)
    # inpaint bg
    mask_bg = tranform_bg(np.ones(img_inpainted.shape))[...,0]
    mask_bg[mask_bg>0] = 1
    bg_inpainted = cv2.inpaint(bg ,1-mask_bg,3,cv2.INPAINT_TELEA)
    # get foreground
    bbox = compute_bbox(mask, random_pad=0)
    img_fg = img[bbox[0]:bbox[1],bbox[2]:bbox[3],:] # * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask_fg = mask[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    # some transform to fg and mask
    tranform_fg = get_transform(img_fg.shape, is_tps=True)
    fg = tranform_fg(img_fg)
    fg_mask = tranform_fg(np.repeat(mask_fg[:, :, np.newaxis], 3, axis=2))[...,0]
    fg_mask[fg_mask>0] = 1
    # random pad fg to the same size of bg
    pad = get_padding(fg.shape, bg.shape)
    fg_pad = pad(fg)
    fg_mask_pad = pad(np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2))[...,0]
    '''
    # get alpha matte by poisson-matting
    # TODO: how to solve the transform?
    alpha = poisson_matte(rgb2gray(img_fg) ,mask_fg)
    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    alpha = pad(tranform_fg(alpha))
    # blend with alpha
    foreground = fg_pad * alpha
    background = bg_inpainted * (1.0 - alpha)
    '''
    # outImage = foreground + background
    test_img = fg_pad * np.repeat(fg_mask_pad[:, :, np.newaxis], 3, axis=2) + bg_inpainted * np.repeat((1-fg_mask_pad)[:, :, np.newaxis], 3, axis=2)
    return test_img.astype(np.uint8), fg_mask_pad.astype(np.uint8)

def main():
    img = cv2.imread('../../video_segmentation/DAVIS2017/trainval/JPEGImages/480p/surf/00000.jpg')
    mask = np.array(Image.open('../../video_segmentation/multi_mask/surf/2/00000.png'))
    a, b = synthesis(img, mask)
    plt.imshow(b)
    plt.show()
    for i in range(10):
        tmp_im, tmp_mask, fg, bg, img_inpainted, image_defect, test_img, olny_fg, mask_bg = synthesis(img, mask.astype(np.uint8))
        # write for watch
        cv2.imwrite(str(i)+'tmp_im.jpg', tmp_im.astype(np.uint8))
        cv2.imwrite(str(i)+'fg.jpg', fg.astype(np.uint8))
        cv2.imwrite(str(i)+'bg.jpg', bg.astype(np.uint8))
        #cv2.imwrite(str(i)+'img_inpainted.jpg', img_inpainted.astype(np.uint8))
        #cv2.imwrite(str(i)+'image_defect.jpg', image_defect.astype(np.uint8))
        cv2.imwrite(str(i)+'test_img.jpg', test_img.astype(np.uint8))
        cv2.imwrite(str(i)+'olny_fg.jpg', olny_fg.astype(np.uint8))
        cv2.imwrite(str(i)+'mask_bg.jpg', mask_bg.astype(np.uint8))

if __name__ == '__main__':
    main()