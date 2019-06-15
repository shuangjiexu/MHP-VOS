import numpy as np
from .utils import clip_around_bbox

def motion_model(imgsize, bbox_history, curridx, numoccframes=0, history_num=10, curbbox=None):
    """
    imgsize is in [h, w]
    bbox_history: array of bbox, in a form of (x1,y1,x2,y2)
    """

    if curridx < 3:
        if curbbox is None or len(curbbox)==0:
            bbox_pred = bbox_history[-1]
        else:
            bbox_pred = curbbox
        return bbox_pred
    
    # predict bbox by history
    # center: [(x,y)]
    center_history = []
    # size: [(w,h)]
    size_history = []
    for bbox in bbox_history:
        center_history.append([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
        size_history.append([bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1])
    center_history = np.array(center_history)
    size_history = np.array(size_history)
    
    # predict velocity
    vel_history = np.diff(center_history, axis=0)
    # (velocity_x, velocity_y)
    vel_predict = np.mean(vel_history[max(len(vel_history)-history_num, 0):,:], axis=0)
    # decay velocity according to numoccframes
    vel_predict = (0.9 ** numoccframes) * vel_predict

    # predict size change
    # size_change_history = np.diff(size_history, axis=0)
    # (velocity_x, velocity_y)
    # size_change_predict = np.mean(size_change_history[max(len(size_change_history)-history_num, 0):], axis=0)


    # predict obj center
    obj_center_pred = center_history[-1] + vel_predict
    obj_size_pred = np.mean(size_history[max(len(size_history)-history_num, 0):,:], axis=0) # size_history[-1] + size_change_predict

    # TODO: smooth
    if curbbox is not None and len(curbbox)!=0:
        halfobjsz = np.array([curbbox[2]-curbbox[0]+1, curbbox[3]-curbbox[1]+1]) / 2.0
        objcenter_old = np.array([curbbox[2]+curbbox[0], curbbox[3]+curbbox[1]]) / 2.0
        obj_center_pred = 0.5 * obj_center_pred + 0.5 * objcenter_old
    else:
        halfobjsz = obj_size_pred/2.0
    # get final bbox
    bbox_pred = [obj_center_pred-halfobjsz, obj_center_pred+halfobjsz]
    # check if the boundary is out of image
    bbox_tmp = [bbox_pred[0][0],bbox_pred[0][1],bbox_pred[1][0],bbox_pred[1][1]]
    bbox_pred = clip_around_bbox(bbox_tmp, imgsize)
    return bbox_pred

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