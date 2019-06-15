import numpy as np 
import os

path_to_lucid = '../data/lucid_dataset'
path_to_save = '../data/lucid_flow'
if not os.path.exists(path_to_save):
	os.mkdir(path_to_save)

file_list = sorted(os.listdir(path_to_lucid))
file_num = len(file_list)
datalist = sorted(os.listdir('../data/DAVIS/JPEGImages/480p'))

with open('./lucidpair.txt', 'w') as f:
	for ii in range(file_num):
		if file_list[ii] not in datalist:
			continue
		print(file_list[ii])
		file_path = os.path.join(path_to_lucid, file_list[ii])
		file_save_path = os.path.join(path_to_save, file_list[ii])
		if not os.path.exists(file_save_path):
			os.mkdir(file_save_path)

		# obj_list = sorted(os.listdir(file_path))
		# obj_num = len(obj_list)

		# for jj in range(obj_num):
		# 	img_path = os.path.join(file_path, obj_list[jj])
			# flow_save_path = os.path.join(file_save_path, obj_list[jj])
			# if not os.path.exists(flow_save_path):
			# 	os.mkdir(flow_save_path)
			
		for kk in range(200):
			img1_path = os.path.join(file_path, str(kk).zfill(5)+'_rgb1.jpg')
			img2_path = os.path.join(file_path, str(kk).zfill(5)+'_rgb2.jpg')

			# flow_path = os.path.join(file_save_path, str(kk).zfill(5)+'.flo')
			flow_path1 = os.path.join(file_save_path, 'flownet2_'+str(kk).zfill(5)+'_'+str(kk+1).zfill(5)+'.flo')
			flow_path2 = os.path.join(file_save_path, 'flownet2_'+str(kk+1).zfill(5)+'_'+str(kk).zfill(5)+'.flo')

			f.write(img1_path+' '+img2_path+' '+flow_path1+'\n')
			f.write(img2_path+' '+img1_path+' '+flow_path2+'\n')

			# f.write(img1_path+' '+img2_path+' '+flow_path+'\n')
		print('load file ' + file_list[ii] + ' Finish!')