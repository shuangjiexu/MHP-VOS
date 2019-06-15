# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from .deeplabv3plus import deeplabv3plus

def generate_net(MODEL_NAME='deeplabv3plus'):
	if MODEL_NAME == 'deeplabv3plus' or MODEL_NAME == 'deeplabv3+':
		return deeplabv3plus()
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%MODEL_NAME)
