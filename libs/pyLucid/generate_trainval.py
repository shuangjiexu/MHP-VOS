import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import *

'''
This is the code for generating multi-mask data on Davis2017_train&val dataset
'''

path_to_dataset = '../data/DAVIS/train'
path_to_save = '../data/multi_mask'
if not os.path.exists(path_to_save):
	os.mkdir(path_to_save)

file_name = sorted(os.listdir(os.path.join(path_to_dataset, 'JPEGImages')))
file_num = len(file_name)

for ii in range(file_num):
	path_to_file = os.path.join(path_to_save, file_name[ii])
	if not os.path.exists(path_to_file):
		os.mkdir(path_to_file)

	img_file = os.path.join(path_to_dataset, 'JPEGImages', file_name[ii])
	mask_file = os.path.join(path_to_dataset, 'Annotations', file_name[ii])
	img_num = len(os.listdir(img_file))

	for jj in range(img_num):
		img_path = os.path.join(img_file, str(jj).zfill(5)+'.jpg')
		mask_path = os.path.join(mask_file, str(jj).zfill(5)+'.png')
		object_num = get_obj_num(mask_path)

		if jj == 0:
			object_num_toal = object_num

		#save origin img
		img = cv2.imread(img_path)
		img_save_path = os.path.join(path_to_file, 'origin')
		if not os.path.exists(img_save_path):
			os.mkdir(img_save_path)
		cv2.imwrite(os.path.join(img_save_path, str(jj).zfill(5)+'.jpg'), img)

		if object_num_toal == object_num:
			for kk in range(object_num):
				if kk == 0:
					continue

				#save mask
				mask_save_path = os.path.join(path_to_file, str(kk))
				if not os.path.exists(mask_save_path):
					os.mkdir(mask_save_path)

				mask, palette = load_mask(mask_path, kk)
				imwrite_index(os.path.join(mask_save_path, str(jj).zfill(5)+'.png'), mask, palette)
		else:
			for kk in range(object_num_toal):
				if kk == 0:
					continue

				#save mask
				mask_save_path = os.path.join(path_to_file, str(kk))
				if not os.path.exists(mask_save_path):
					os.mkdir(mask_save_path)

				mask, palette = load_mask(mask_path, kk)
				imwrite_index(os.path.join(mask_save_path, str(jj).zfill(5)+'.png'), mask, palette)