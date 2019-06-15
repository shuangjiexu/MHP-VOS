import os
import numpy as np
from math import sin, cos, pi, sqrt
from PIL import Image 
import skimage.measure as measure

def valid_mask(mask, thr=0.3):
    # write origion mask
    mask_valid = mask.copy()
    # print(mask_valid.max())
    mask_valid[mask_valid>thr] = 1
    mask_valid[mask_valid<=thr] = 0
    return mask_valid.astype(np.float32)

def imwrite_index(filename,array,color_palette):
    """ Save indexed png."""

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im =im.convert("P")
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

def get_obj_img(config, sequence):
    path = os.path.join(config.mask_path, sequence, '00000.png')
    obj_num = get_obj_num(path)
    img_list = []
    for i in range(obj_num):
        mask = load_mask(path, i+1)
        bbox = compute_bbox_from_mask(mask)
        img_list.append(bbox)
    return os.path.join(config.img_dir, sequence, '00000.jpg'), img_list

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

def get_obj_num(path):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    ###
    mask = mask.copy()
    # mask[mask>0] = 1
    ###
    obj_num = len(np.unique(mask))
    return obj_num-1

def load_mask(path, obj_id):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    mask = mask.copy()
    ###
    # mask[mask>0] = 1
    ###
    mask[mask!=obj_id] = 0
    mask[mask!=0] = 1
    return mask.astype(np.float32)

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width]. Mask pixels are either 1 or 0.

    Returns: bbox array [x1, y1, x2, y2].
    """
    # Bounding box.
    mask = mask.astype(np.int)
    regions = measure.regionprops(mask)
    if len(regions):
        y1,x1,y2,x2 = regions[0].bbox 
        bbox = np.array([x1, y1, x2, y2])
        return bbox.astype(np.int32)
    else:
        return np.array([0,0,0,0])

def eigsorted(cov):
    """Return eigenvalues, sorted."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

def cov_ellipse(cov, nstd):
    """Get the covariance ellipse."""
    vals, vecs = eigsorted(cov)
    r1, r2 = nstd * np.sqrt(vals)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    return r1, r2, theta

def gaussian_bbox(x, P, nstd=2):
    """Return boudningbox for gaussian."""
    r1, r2, theta = cov_ellipse(P, nstd)
    ux = r1 * cos(theta)
    uy = r1 * sin(theta)
    vx = r2 * cos(theta + pi/2)
    vy = r2 * sin(theta + pi/2)

    dx = sqrt(ux*ux + vx*vx)
    dy = sqrt(uy*uy + vy*vy)

    return (float(x[0] - dx),
            float(x[0] + dx),
            float(x[1] - dy),
            float(x[1] + dy))

def bbox_from_roi(im_size, roi, rate):
    """ get large bbox from roi
    roi is in [y1,x1,y2,x2]
    imgsize is in [h, w]
    """
    bbox = [roi[1], roi[0], roi[3], roi[2]] # [x1,y1,x2,y2]
    x_change = int((bbox[2]-bbox[0])*rate/2)
    y_change = int((bbox[3]-bbox[1])*rate/2)
    bbox = clip_around_bbox([bbox[0]-x_change, bbox[1]-y_change, bbox[2]+x_change, bbox[3]+y_change], im_size)
    return bbox

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

def calc_bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

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

def fuse_mask(cfgs, mask1, mask2):
    mask = np.zeros(mask1.shape)
    mask = mask1+mask2
    mask[mask>cfgs.VALID_TH] = 1
    mask[mask<=cfgs.VALID_TH] = 0
    return mask

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