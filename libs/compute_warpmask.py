import numpy as np 
import os
import cv2

from utils import *
import matplotlib.pyplot as plt

def warpFL(im2, flow):
    h, w = flow.shape[:2]
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    # print(flow[:,:,1].shape)
    warpI2 = cv2.remap(im2, flow, None, cv2.INTER_CUBIC, borderMode =cv2.BORDER_CONSTANT)
    warpI2[np.isnan(warpI2)] = 0
    
    return warpI2

def warpFLColor(im1, im2, flow):
    h, w, c = im1.shape
    warpI2 = np.zeros((h,w,c))
    flow_old = flow.copy()
    for i in range(c):
        flow = flow_old.copy()
        # plt.imshow(flow[:,:,0])
        # plt.show()
        im = warpFL(im2[:,:,i],flow)

        warpI2[:,:,i] = im
    return warpI2

path_to_flow = '../data/lucid_flow'
seg_list = sorted(os.listdir(path_to_flow))

# datalist = ['bmx-trees', 'horsejump-high', 'kite-surf', 'motocross-jump', 'paragliding-launch',
#             'scooter-black', 'soapbox']

for seg_name in seg_list:
    # if seg_name not in datalist:
    #     continue
    print('do for '+seg_name)
    path_to_file = os.path.join(path_to_flow, seg_name)
    # frame_length = len(os.listdir('../DAVIS2017/trainval/JPEGImages/480p/'+seg_name)) - 1

    for frame in range(200):
        im1_id = frame
        # print im1_id
        im2_id = frame
        # print('../DAVIS2017/test_dev/JPEGImages/480p/'+seg_name+'/%05d.jpg'%im1_id)
        im1 = cv2.imread('./data/lucid_dataset/'+seg_name+'/%05d_rgb1.jpg'%im1_id)
        im2 = cv2.imread('./data/lucid_dataset/'+seg_name+'/%05d_rgb2.jpg'%im2_id)

        flow_01_file = os.path.join(path_to_file, 'flownet2_%05d_%05d.flo'%(im1_id, im2_id+1))  # flownet2_00070_00069.flo
        flow_10_file = os.path.join(path_to_file, 'flownet2_%05d_%05d.flo'%(im2_id+1, im1_id)) 
        warp_01_file = os.path.join(path_to_file, 'flownet2_%05d_%05d.png'%(im1_id, im2_id+1))  # flownet2_00070_00069.png
        warp_10_file = os.path.join(path_to_file, 'flownet2_%05d_%05d.png'%(im2_id+1, im1_id))

        flow01 = read_flow(flow_01_file)
        flow10 = read_flow(flow_10_file)

        warpI12 = warpFLColor(im1,im2,flow01);
        warpI21 = warpFLColor(im2,im1,flow10);
        
        cv2.imwrite(warp_01_file, warpI12);
        cv2.imwrite(warp_10_file, warpI21);
    print('Done!')

# path_to_flow = '../trainval_flow'
# seg_list = sorted(os.listdir(path_to_flow))

# for seg_name in seg_list:
#     print('do for '+seg_name)
#     path_to_file = os.path.join(path_to_flow, seg_name)
#     frame_length = len(os.listdir('../DAVIS2017/trainval/JPEGImages/480p/'+seg_name)) - 1

#     for frame in range(frame_length):
#         im1_id = frame
#         # print im1_id
#         im2_id = frame + 1
#         # print('../DAVIS2017/test_dev/JPEGImages/480p/'+seg_name+'/%05d.jpg'%im1_id)
#         im1 = cv2.imread('../DAVIS2017/trainval/JPEGImages/480p/'+seg_name+'/%05d.jpg'%im1_id)
#         im2 = cv2.imread('../DAVIS2017/trainval/JPEGImages/480p/'+seg_name+'/%05d.jpg'%im2_id)

#         flow_01_file = os.path.join(path_to_file, 'flownet2_%05d_%05d.flo'%(im1_id, im2_id))  # flownet2_00070_00069.flo
#         flow_10_file = os.path.join(path_to_file, 'flownet2_%05d_%05d.flo'%(im2_id, im1_id)) 
#         warp_01_file = os.path.join(path_to_file, 'flownet2_%05d_%05d.png'%(im1_id, im2_id))  # flownet2_00070_00069.png
#         warp_10_file = os.path.join(path_to_file, 'flownet2_%05d_%05d.png'%(im2_id, im1_id))

#         flow01 = read_flow(flow_01_file)
#         flow10 = read_flow(flow_10_file)

#         warpI12 = warpFLColor(im1,im2,flow01);
#         warpI21 = warpFLColor(im2,im1,flow10);
        
#         cv2.imwrite(warp_01_file, warpI12);
#         cv2.imwrite(warp_10_file, warpI21);
#     print('Done!')