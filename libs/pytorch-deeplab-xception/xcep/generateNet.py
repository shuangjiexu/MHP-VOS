# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from xcep.deeplabv3plus import deeplabv3plus
# from net.supernet import SuperNet
# from net.EANet import EANet
# from net.DANet import DANet
# from net.deeplabv3plushd import deeplabv3plushd
# from net.DANethd import DANethd
def generate_net(MODEL_NAME='deeplabv3plus'):
	if MODEL_NAME == 'deeplabv3plus' or MODEL_NAME == 'deeplabv3+':
		return deeplabv3plus()
	# if cfg.MODEL_NAME == 'supernet' or cfg.MODEL_NAME == 'SuperNet':
	# 	return SuperNet(cfg)
	# if cfg.MODEL_NAME == 'eanet' or cfg.MODEL_NAME == 'EANet':
	# 	return EANet(cfg)
	# if cfg.MODEL_NAME == 'danet' or cfg.MODEL_NAME == 'DANet':
	# 	return DANet(cfg)
	# if cfg.MODEL_NAME == 'deeplabv3plushd' or cfg.MODEL_NAME == 'deeplabv3+hd':
	# 	return deeplabv3plushd(cfg)
	# if cfg.MODEL_NAME == 'danethd' or cfg.MODEL_NAME == 'DANethd':
	# 	return DANethd(cfg)
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%MODEL_NAME)
