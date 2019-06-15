import os

import numpy as np
import cv2

def read_flow(flowfile):
    f = open(flowfile, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)

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

def main():
    img_dir = '/data1/shuangjiexu/data/DAVIS_2017/JPEGImages/480p/motorbike'
    flow_dir = '/data1/shuangjiexu/data/DAVIS_2017/Results/Segmentations/480p/opticalflow_flownet2/motorbike'
    mask_dir = '/data1/shuangjiexu/data/DAVIS_2017/Annotations/480p/motorbike'
    im1_id = 36
    im2_id = 37
    obj_id = 1

    from davis import io
    mask ,_ = io.imread_indexed(os.path.join(mask_dir, '%05d.png'%im1_id))
    mask_1 ,_ = io.imread_indexed(os.path.join(mask_dir, '%05d.png'%im2_id))
    mask_tmp = mask.copy()
    mask_tmp[mask_tmp != obj_id] = 0
    mask_tmp[mask_tmp != 0] = 1

    mask_write_1 = mask_tmp.copy()
    mask_write_1[mask_write_1 > 0.3]=255
    cv2.imwrite('mask.jpg', mask_write_1)
    
    warped_mask, validflowmap01,_,_ = warp_mask(mask_tmp, im1_id, im2_id, flow_dir, img_dir)
    print(warped_mask.shape)

    mask_write_2 = warped_mask.copy()
    mask_write_2[mask_write_2 > 0.3]=255
    cv2.imwrite('mask_warped.jpg', mask_write_2)

if __name__ == '__main__':
    main()
