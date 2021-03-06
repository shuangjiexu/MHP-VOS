import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import *

'''
This is the code for generating multi-mask data on Davis2017_train&val dataset
'''

# path_to_dataset = '../all_test_file/test_lucid_dataset'
# path_to_save = '../all_test_file/lucid_mask'
path_to_dataset = '../data/lucid_dataset'
path_to_save = '../data/lucid_mask'
if not os.path.exists(path_to_save):
	os.mkdir(path_to_save)

file_name = sorted(os.listdir(path_to_dataset))
file_num = len(file_name)

# datalist = ['bmx-trees', 'horsejump-high', 'kite-surf', 'motocross-jump', 'paragliding-launch',
#             'scooter-black', 'soapbox']

datalist = sorted(os.listdir('../data/DAVIS/JPEGImages/480p'))

for ii in range(file_num):
	if file_name[ii] not in datalist:
		continue

	print(file_name[ii])
	path_to_file = os.path.join(path_to_save, file_name[ii])
	if not os.path.exists(path_to_file):
		os.mkdir(path_to_file)

	img_file = os.path.join(path_to_dataset, file_name[ii])
	mask_file = os.path.join(path_to_dataset, file_name[ii])
	img_num = 200

	for jj in range(img_num):
		img_path = os.path.join(img_file, str(jj).zfill(5)+'_rgb2.jpg')
		mask1_path = os.path.join(mask_file, str(jj).zfill(5)+'_gt1.png')
		mask2_path = os.path.join(mask_file, str(jj).zfill(5)+'_gt2.png')
		object_num = get_obj_num(mask1_path)
		if jj == 0:
			object_num_toal = object_num
		if object_num > object_num_toal:
			object_num_toal = object_num

	for jj in range(img_num):
		img_path = os.path.join(img_file, str(jj).zfill(5)+'_rgb2.jpg')
		mask1_path = os.path.join(mask_file, str(jj).zfill(5)+'_gt1.png')
		mask2_path = os.path.join(mask_file, str(jj).zfill(5)+'_gt2.png')
		object_num = get_obj_num(mask1_path)


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

				mask1, palette1 = load_mask(mask1_path, kk)
				imwrite_index(os.path.join(mask_save_path, str(jj).zfill(5)+'_gt1.png'), mask1, palette1)
				mask2, palette2 = load_mask(mask2_path, kk)
				imwrite_index(os.path.join(mask_save_path, str(jj).zfill(5)+'_gt2.png'), mask2, palette2)
		else:
			for kk in range(object_num_toal):
				if kk == 0:
					continue

				#save mask
				mask_save_path = os.path.join(path_to_file, str(kk))
				if not os.path.exists(mask_save_path):
					os.mkdir(mask_save_path)

				mask1, palette1 = load_mask(mask1_path, kk)
				imwrite_index(os.path.join(mask_save_path, str(jj).zfill(5)+'_gt1.png'), mask1, palette1)
				mask2, palette2 = load_mask(mask2_path, kk)
				imwrite_index(os.path.join(mask_save_path, str(jj).zfill(5)+'_gt2.png'), mask2, palette2)