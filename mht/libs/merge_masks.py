import numpy as np
import cv2
import os

import skimage.measure as measure

from libs.davis import *
from libs.warp_mask import warp_mask

def generate_gaussian_map_bbox(bbox, img_size):
    """
    bbox: [x1, y1, x2, y2]
    img_size: [h,w]
    missed_frame_num: int. number of frames missed
    """
    h,w = img_size[:]
    center_x = (bbox[0]+bbox[2])/2
    center_y = (bbox[1]+bbox[3])/2
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x - center_x
    y = y - center_y
    sigma_x = (bbox[2]-bbox[0]) / 2.0
    sigma_y = (bbox[3]-bbox[1]) / 2.0
    gaussian_map = np.exp(-(np.power(x,2)/(2*(sigma_x*sigma_x)) + np.power(y,2)/(2*sigma_y*sigma_y)))
    return gaussian_map

def merge_masks(cfgs, masks, maskres_prev, bboxs, imgidx, seqname):
    """
    compute final mask with format: [0,1,2,3...] as different class
    fix overlap between different single object mask
    masks: mask list
    prevcnnresp: this is the original mask from network
    """
    mask_shape = masks[0].shape
    assert len(mask_shape) == 2
    # generate gaussian map
    gaussian_maps = []
    for bbox in bboxs:
        gaussian_maps.append(generate_gaussian_map_bbox(bbox, mask_shape))
    maskres = np.zeros(mask_shape)
    cntoverlap = np.zeros((len(masks),)+mask_shape)
    overbitmap = np.zeros((len(masks),)+mask_shape)
    for obj_id in range(1,len(masks)+1):
        maskres[masks[obj_id-1]>cfgs.VALID_TH] = obj_id
        # check overlapped masks
        cntoverlap[obj_id-1][masks[obj_id-1]>cfgs.VALID_TH] = 1
        # bitmapobj(prevmask{obj_id} > binmask_thresh) = bitset(bitmapobj(prevmask{obj_id} > binmask_thresh), obj_id)
        overbitmap[obj_id-1][masks[obj_id-1]>cfgs.VALID_TH] = 2**obj_id
    overlapmap = (np.sum(cntoverlap,axis=0) >= 2).astype(np.int)
    overlapmap = mask_remove_small_blobs(cfgs, overlapmap)
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
                imgres = masks[obj_id-1] * gaussian_maps[obj_id-1] * subbwmap
                sumcnnresps.append(np.sum(imgres))
            sortedidx = sorted(range(1, len(sumcnnresps)+1), key=lambda k: sumcnnresps[k-1], reverse = True)
            if sumcnnresps[sortedidx[0]-1] * 0.5 > sumcnnresps[sortedidx[1]-1]:
                # we are confident
                idxsel = sortedidx[0]
            else:
                if imgidx > 1:
                    # very ambiguous, choose the one with temporal coherency
                    maskprev_obj1 = (maskres_prev == sortedidx[0]).astype(np.double)
                    maskprev_obj2 = (maskres_prev == sortedidx[1]).astype(np.double)
                    warped_mask1, warped_validmap1, _, _ = warp_mask(maskprev_obj1, imgidx-1, imgidx, 
                                                                    os.path.join(cfgs.flow_dir, seqname), 
                                                                    os.path.join(cfgs.img_dir, seqname))
                    warped_mask2, warped_validmap2, _, _ = warp_mask(maskprev_obj2, imgidx-1, imgidx, 
                                                                    os.path.join(cfgs.flow_dir, seqname), 
                                                                    os.path.join(cfgs.img_dir, seqname))
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

