from patchPaint import paint
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lucidDream import dreamData
from utils import *

'''
This is the code for generating lucid data on Davis2017_val dataset
'''

path_to_dataset = '../data/DAVIS'
# print(os.listdir(path_to_dataset)) 
# path_forlucid = path_to_dataset + '/ImageSets/480p/val.txt'

datalist = sorted(os.listdir('../data/DAVIS/JPEGImages/480p'))

# with open(path_forlucid) as f:
#     seqs = f.readlines()
    # print(seqs)
img_list = []
labels = []

for seq in datalist:
    images = np.sort(os.listdir(os.path.join(path_to_dataset, 'JPEGImages/480p/', seq.strip())))
    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
    img_list.append(images_path[0])
    lab = np.sort(os.listdir(os.path.join(path_to_dataset, 'Annotations/480p/', seq.strip())))
    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
    labels.append(lab_path[0])

print('generate lucid data for '+str(len(img_list))+' videos begin!')

path_to_save = '../data/lucid_dataset'
if not os.path.exists(path_to_save):
	os.mkdir(path_to_save)


num_file = len(img_list)

for ii in range(num_file):
	# if ii <  22:
	# 	continue
	file = img_list[ii]
	img_path = path_to_dataset + '/' + file
	label_path = path_to_dataset + '/' + labels[ii]
	file_name = file.split('/')[2]
	print(file_name)
	if not os.path.exists(path_to_save + '/' + file_name):
		os.mkdir(path_to_save + '/' + file_name)

	object_num = get_obj_num(label_path)
	# for jj in range(object_num):
	# 	if jj == 0:
	# 		continue
	path_to_jj = os.path.join(path_to_save, file_name)
		# mask, palette = load_mask(label_path, jj)
		# mask = Image.fromarray(mask, 'P')
		# mask.save(os.path.join(path_to_jj, 'mask.png'))
		# # mask = Image.open(os.path.join(path_to_jj, 'mask.png'))
		# # print np.array(mask)[300:500,300:500]
		# bb
		# if not os.path.exists(path_to_jj):
		# 	os.mkdir(path_to_jj)

	generate_num = 200
	Iorg=cv2.imread(img_path)
	mask = Image.open(label_path)
	palette=mask.getpalette()

	mask = np.array(mask)
	print(mask.max())
	bg=paint(Iorg,mask,False)

	for kk in range(generate_num):
		path_rgb1_save = os.path.join(path_to_jj, str(kk).zfill(5)+'_rgb1.jpg')
		path_rgb2_save = os.path.join(path_to_jj, str(kk).zfill(5)+'_rgb2.jpg')
		path_mask1_save = os.path.join(path_to_jj, str(kk).zfill(5)+'_gt1.png')
		path_mask2_save = os.path.join(path_to_jj, str(kk).zfill(5)+'_gt2.png')
		path_flow_save = os.path.join(path_to_jj, str(kk).zfill(5)+'_flow.png')

		im_1,gt_1,bb_1,im_2,gt_2,bb_2,fb,ff=dreamData(Iorg,mask,bg,True)

		# # Image 1 in this pair.
		cv2.imwrite(path_rgb1_save, im_1)

		# Mask for image 1.
		gtim1=Image.fromarray(gt_1,'P')
		gtim1.putpalette(palette)
		gtim1.save(path_mask1_save)

		# # Deformed previous mask for image 1.
		# bbim1=Image.fromarray(bb_1,'P')
		# bbim1.putpalette(palette)
		# bbim1.save('gen1bb.png')

		# Image 2 in this pair.
		cv2.imwrite(path_rgb2_save, im_2)

		# Mask for image 2.
		gtim2=Image.fromarray(gt_2,'P')
		gtim2.putpalette(palette)
		gtim2.save(path_mask2_save)

		# # Deformed previous mask for image 2.
		# bbim2=Image.fromarray(bb_2,'P')
		# bbim2.putpalette(palette)
		# bbim2.save('gen2bb.png')

		# # Optical flow from Image 2 to Image 1. 
		# # Its magnitude can be used as a guide to get mask of Image 2.
		# flowmag=np.sqrt(np.sum(fb**2,axis=2))
		# flowmag_norm=(flowmag-flowmag.min())/(flowmag.max()-flowmag.min())
		# flowmagim=(flowmag_norm*255+0.5).astype('uint8')
		# flowim=Image.fromarray(flowmagim,'L')
		# flowim.save('gen2fb.png')

		# Optical flow from Image 1 to Image 2. 
		# Its magnitude can be used as a guide to get mask of Image 1.
		flowmag=np.sqrt(np.sum(ff**2,axis=2))
		flowmag_norm=(flowmag-flowmag.min())/(flowmag.max()-flowmag.min())
		flowmagim=(flowmag_norm*255+0.5).astype('uint8')
		flowim=Image.fromarray(flowmagim,'L')
		flowim.save(path_flow_save)
	print('generate lucid for ' + file_name + ' finish!')