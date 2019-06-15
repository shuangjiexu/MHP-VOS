import numpy as np 
import os 
from PIL import Image

from utils import *

path_to_singlemask = '/path/to/results'
path_to_color = '../DAVIS2017/test_dev'
path_to_fusemask = './final_fused'
if not os.path.exists(path_to_fusemask):
    os.mkdir(path_to_fusemask)

seq_list = []
with open(path_to_color+'/ImageSets/2017/test-dev.txt') as f:
    seq = f.readlines()
for seq_name in seq:
    seq_list.append(seq_name.strip())

for seq_name in seq_list:
    path_to_save = os.path.join(path_to_fusemask, seq_name)
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    path_to_file = os.path.join(path_to_singlemask, seq_name)

    obj_num = len(os.listdir(path_to_file))
    video_length = len(os.listdir(os.path.join(path_to_file, str(1))))

    annotation_path = os.path.join(path_to_color, 'Annotations', '480p', seq_name, '00000.png')
    maskres_prev, palette = imread_indexed(annotation_path)

    for frame in range(video_length):
        frame += 1
        masks = []

        for obj_id in range(obj_num):
            obj_id += 1

            path_to_mask = os.path.join(path_to_file, str(obj_id), str(frame).zfill(5)+'.png')
            mask = load_mask(path_to_mask, 255)
            masks.append(mask)
        maskres = merge_masks(masks, maskres_prev, frame, seqname)
        maskres_prev = maskres
        