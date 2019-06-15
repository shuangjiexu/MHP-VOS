import numpy as np
from PIL import Image
import skimage.measure as measure
import cv2
import scipy.signal as signal

def get_obj_num(path):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    obj_num = len(np.unique(mask))
    return obj_num

def is_obj_too_small(mask):
    if np.sum(mask>0.5) < 30*30:
        return 1
    else:
        return 0

def imwrite_index(filename,array,color_palette):
    """ Save indexed png."""

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im =im.convert("P")
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

def load_mask(path, obj_id):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    mask = mask.copy()
    mask[mask!=obj_id] = 0
    mask[mask!=0] = 1
    return mask.astype(np.float32)

def valid_mask(cfgs, mask):
    # write origion mask
    mask_valid = mask.copy()
    mask_valid[mask_valid>cfgs.VALID_TH] = 1
    mask_valid[mask_valid<=cfgs.VALID_TH] = 0
    return mask_valid.astype(np.float32)

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

def get_area_from_bbox(bbox):
    """
    bbox: [x1,y1,x2,y2]

    return: area of bbox 
    """
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]
    return int(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def valid_area(mask):
    return np.sum((mask>0).astype(np.int))


def generate_gaussian_map(cfgs, bbox, img_size, missed_frame_num, max_num=None):
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
    if max_num:
        sigma_x = (bbox[2]-bbox[0])*(cfgs.DILATION_COEFFICIENT ** min(max_num,missed_frame_num)) / 2.0
        sigma_y = (bbox[3]-bbox[1])*(cfgs.DILATION_COEFFICIENT ** min(max_num,missed_frame_num)) / 2.0
    else:
        sigma_x = (bbox[2]-bbox[0])*(cfgs.DILATION_COEFFICIENT ** missed_frame_num) / 2.0
        sigma_y = (bbox[3]-bbox[1])*(cfgs.DILATION_COEFFICIENT ** missed_frame_num) / 2.0
    gaussian_map = np.exp(-(np.power(x,2)/(2*(sigma_x*sigma_x)) + np.power(y,2)/(2*sigma_y*sigma_y)))
    return gaussian_map

def mask_remove_small_blobs(cfgs, mask):
    labeled_mask = measure.label((mask>0.5).astype(np.int))
    regions = measure.regionprops(labeled_mask)
    if len(regions):
        mask_out = np.zeros(mask.shape)
        regions.sort(key=lambda x:(x.area)) 
        for region in regions:
            if region.area > cfgs.MIN_PIXEL_NUMBER:
                mask_out[region.coords[:,0], region.coords[:,1]] = 1
        return mask_out.astype(np.float)
    else:
        return mask.copy().astype(np.float)

def merge_overlapped_blobs(cfgs, mask1, mask2):
    overlapmap = np.logical_and(mask1>cfgs.OVERLAP_TH, mask2>cfgs.OVERLAP_TH).astype(np.int)
    if np.sum(overlapmap):
        mask_out = mask1.copy()
        labed_mask2 = measure.label(mask2)
        bw2m = labed_mask2 * overlapmap
        idxs = np.unique(bw2m)
        for region_id in idxs:
            if region_id != 0:
                mask_out[labed_mask2==region_id] = 1
        return mask_out
    else:
        return mask1.copy()

def max_blob_pixel_num(mask):
    labeled_mask = measure.label((mask>0).astype(np.int))
    regions = measure.regionprops(labeled_mask)
    if len(regions):
        regions.sort(key=lambda x:(x.area)) 
        return regions[-1].area
    else:
        return 0

def is_mask_clutter(cfgs, mask):
    """
    return isclutter
    """
    isclutter = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img_erosion = cv2.erode(mask, kernel, iterations=1)
    maskthin = mask_remove_small_blobs(cfgs, img_erosion)
    img_dilation = cv2.dilate(maskthin, kernel, iterations=1)

    numpixbefore = np.sum((mask>0).astype(np.int))
    numpixafter = np.sum((img_dilation>0).astype(np.int))

    if numpixbefore == 0:
        isclutter = 0
        return isclutter

    pixratio = (numpixafter+0.0) / numpixbefore
    if pixratio < 0.8:
        isclutter = 1
    else:
        isclutter = 0
    return isclutter

def compute_bbox_from_mask(cfgs, mask):
    # return [x1, y1, x2, y2]
    margin = cfgs.PREDICT_BBOX_MARGIN
    h,w = mask.shape[:2]
    # [x1, y1, x2, y2]
    bbox = extract_bboxes((mask>0.1).astype(np.int16))
    h_box = bbox[3] - bbox[1]
    w_box = bbox[2] - bbox[0]
    # apply margin
    bbox_margined = [bbox[0]-margin*w_box, bbox[1]-margin*h_box, bbox[2]+margin*w_box, bbox[3]+margin*h_box]
    bbox_margined = clip_around_bbox(bbox_margined, mask.shape[:2])
    return bbox_margined

def compute_bbox_from_mask_withsize(cfgs, mask, smoothed_hist_bbox_size):
    margin = cfgs.PREDICT_BBOX_MARGIN
    h,w = mask.shape[:2]
    # [x1, y1, x2, y2]
    bbox = extract_bboxes((mask>0.1).astype(np.int16))
    h_box = bbox[3] - bbox[1]
    w_box = bbox[2] - bbox[0]
    # apply margin
    bbox_margined = [bbox[0]-margin*w_box, bbox[1]-margin*h_box, bbox[2]+margin*w_box, bbox[3]+margin*h_box]
    bbox_margined = clip_around_bbox(bbox_margined, mask.shape[:2])
    # smooth  [h,w]
    cursz = np.array([bbox_margined[3]-bbox_margined[1]+1, bbox_margined[2]-bbox_margined[0]+1]).astype(np.float)
    targetsz = smoothed_hist_bbox_size
    half_diffsz = np.round((targetsz - cursz)/2)
    bbox = [bbox_margined[0]-half_diffsz[1], bbox_margined[1]-half_diffsz[0], bbox_margined[2]+half_diffsz[1], bbox_margined[3]+half_diffsz[0]]
    bbox = clip_around_bbox(bbox, mask.shape[:2])
    return bbox

def predict_bbox_bymaskflow(imgsize, flow12, validflowmap12, prevmask, prevbbox, curbbox):
    """
    prevbbox is the bbox of last frame
    curbbox is detected in current frame
    prevmask is the mask of last frame
    flow12 is flow from last frame to current frame
    """
    # erode valid map, only leave confident pixels
    validflowmap12 = signal.medfilt2d(validflowmap12, (3,3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    validflowmap12 = cv2.erode(validflowmap12, kernel, iterations=1)
    validpix = np.sum(validflowmap12 * prevmask)
    # when there is no reliable flow, trust current bbox? 
    if validpix == 0:
        return curbbox

    flow12forobj = flow12 * np.stack([validflowmap12]*2,axis=2) * np.stack([prevmask]*2,axis=2)
    sumdx = np.sum(flow12forobj[...,0])
    sumdy = np.sum(flow12forobj[...,1])
    meandxdy = np.array([sumdx,sumdy]).astype(np.float) / (validpix+0.0)

    prevobjcenter = np.array([(prevbbox[0]+prevbbox[2])/2, (prevbbox[1]+prevbbox[3])/2]).astype(np.float)
    curobjcenter = np.array([(curbbox[0]+curbbox[2])/2, (curbbox[1]+curbbox[3])/2]).astype(np.float)
    diff = curobjcenter - prevobjcenter

    prevobjsz = np.array([prevbbox[2]-prevbbox[0]+1, prevbbox[3]-prevbbox[1]+1]).astype(np.float)
    # if mean flow value is consistent with prevbbox->curbbox, accept curbbox
    # otherwise compute a mean from prevbbox to curbbox
    if np.linalg.norm(diff-meandxdy) < max(np.linalg.norm(prevobjsz)*0.1, 20): # to handle small objects, use a min 20 thresh
        bbox_pred = curbbox
    else:
        halfobjsz = np.array([curbbox[2]-curbbox[0]+1, curbbox[3]-curbbox[1]+1]).astype(np.float) / 2.0
        objcenter_pred = prevobjcenter + meandxdy
        objcenter_pred = 0.5 * objcenter_pred + 0.5 * curobjcenter
        bbox_pred = [objcenter_pred-halfobjsz, objcenter_pred+halfobjsz]
        bbox_pred = [bbox_pred[0][0],bbox_pred[0][1],bbox_pred[1][0],bbox_pred[1][1]]
        bbox_pred = clip_around_bbox(bbox_pred, imgsize)
    return bbox_pred

def calc_mask_iou(curobjmask, warped_mask):
    curbw = (curobjmask>0.5).astype(np.int16)
    warpbw = (warped_mask>0.5).astype(np.int16)

    maskunion = np.logical_or(curbw,warpbw)
    maskinter = np.logical_and(curbw,warpbw)

    cntunion = np.sum(maskunion)+0.0
    cntinter = np.sum(maskinter)+0.0
    if cntunion == 0:
        return 0
    else:
        return cntinter/cntunion
