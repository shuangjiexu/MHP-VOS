import numpy as np 
import os 
from PIL import Image
import cv2

import skimage.measure as measure


def imread_indexed(filename):
    """ Load image given filename."""

    im = Image.open(filename)

    annotation = np.atleast_3d(im)[...,0]
    return annotation,np.array(im.getpalette()).reshape((-1,3))

def load_mask(path, obj_id):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    mask = mask.copy()
    mask[mask!=obj_id] = 0
    mask[mask!=0] = 1
    return mask.astype(np.float32)

def read_flow(flowfile):
    f = open(flowfile, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)

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

def mask_remove_small_blobs(mask):
    labeled_mask = measure.label((mask>0.5).astype(np.int))
    regions = measure.regionprops(labeled_mask)
    if len(regions):
        mask_out = np.zeros(mask.shape)
        regions.sort(key=lambda x:(x.area)) 
        for region in regions:
            if region.area > 20*20:
                mask_out[region.coords[:,0], region.coords[:,1]] = 1
        return mask_out.astype(np.float)
    else:
        return mask.copy().astype(np.float)

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


def merge_masks(masks, maskres_prev, imgidx, seqname):
    """
    compute final mask with format: [0,1,2,3...] as different class
    fix overlap between different single object mask
    masks: mask list
    prevcnnresp: this is the original mask from network
    """
    mask_shape = masks[0].shape
    assert len(mask_shape) == 2
    maskres = np.zeros(mask_shape)
    cntoverlap = np.zeros((len(masks),)+mask_shape)
    overbitmap = np.zeros((len(masks),)+mask_shape)
    for obj_id in range(1,len(masks)+1):
        maskres[masks[obj_id-1]==1] = obj_id
        # check overlapped masks
        cntoverlap[obj_id-1][masks[obj_id-1]==1] = 1
        # bitmapobj(prevmask{obj_id} > binmask_thresh) = bitset(bitmapobj(prevmask{obj_id} > binmask_thresh), obj_id)
        overbitmap[obj_id-1][masks[obj_id-1]==1] = 2**obj_id
    overlapmap = (np.sum(cntoverlap,axis=0) >= 2).astype(np.int)
    overlapmap = mask_remove_small_blobs(overlapmap)
    overlapbitmap = overbitmap * overlapmap

    overlap_objs = np.sum(overlapbitmap,axis=0)
    regions = np.unique(overlap_objs)
    regions = np.delete(regions, 0)
    for region_id in regions:
        region_overlap = (overlap_objs == region_id).astype(np.int)
        labeled_overlap = measure.label(region_overlap)
        obj_regions = measure.regionprops(labeled_overlap)
        for region in obj_regions:
            subbwmap = np.zeros(region_overlap.shape)
            subbwmap[region.coords[:,0], region.coords[:,1]] = 1
            # statistics the regions' possibility and sort
            sumcnnresps = []
            for obj_id in range(1,len(masks)+1):
                # calculate sum of region
                imgres = masks[obj_id-1] * subbwmap
                sumcnnresps.append(np.sum(imgres))
            sortedidx = sorted(range(1, len(sumcnnresps)+1), key=lambda k: sumcnnresps[k-1], reverse = True)
            if sumcnnresps[sortedidx[0]-1] * 0.8 > sumcnnresps[sortedidx[1]-1]:
                # we are confident
                idxsel = sortedidx[0]
            else:
                if imgidx > 1:
                    # very ambiguous, choose the one with temporal coherency
                    maskprev_obj1 = (maskres_prev == sortedidx[0]).astype(np.double)
                    maskprev_obj2 = (maskres_prev == sortedidx[1]).astype(np.double)
                    warped_mask1, warped_validmap1, _, _ = warp_mask(maskprev_obj1, imgidx-1, imgidx, 
                                                                    os.path.join('./test_flow', seqname), 
                                                                    os.path.join('../DAVIS2017/test_dev/JPEGImages/480p', seqname))
                    warped_mask2, warped_validmap2, _, _ = warp_mask(maskprev_obj2, imgidx-1, imgidx, 
                                                                    os.path.join('./test_flow', seqname), 
                                                                    os.path.join('../DAVIS2017/test_dev/JPEGImages/480p', seqname))
                    sum1 = np.sum(subbwmap * warped_mask1 * warped_validmap1)
                    sum2 = np.sum(subbwmap * warped_mask2 * warped_validmap2)
                    if sum1 >= sum2:
                        idxsel = sortedidx[0]
                    else:
                        idxsel = sortedidx[1]
                else:
                    idxsel = sortedidx[0]
            maskres[subbwmap>0] = idxsel
    return maskres