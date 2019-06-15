# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from xcep.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from xcep.backbone import build_backbone
from xcep.ASPP import ASPP

class deeplabv3plus(nn.Module):
	def __init__(self):
		super(deeplabv3plus, self).__init__()
		self.MODEL_BACKBONE = 'xception'
		self.MODEL_ASPP_OUTDIM = 256
		self.MODEL_OUTPUT_STRIDE = 16
		self.MODEL_SHORTCUT_DIM = 48
		self.MODEL_SHORTCUT_KERNEL = 1
		self.MODEL_NUM_CLASSES = 1

		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=self.MODEL_ASPP_OUTDIM, 
				rate=16//self.MODEL_OUTPUT_STRIDE)
#		self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=self.MODEL_OUTPUT_STRIDE//4)

		indim = 256
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, self.MODEL_SHORTCUT_DIM, self.MODEL_SHORTCUT_KERNEL, 1, padding=self.MODEL_SHORTCUT_KERNEL//2),
				nn.ReLU(inplace=True),		
		)		
		self.cat_conv = nn.Sequential(
				nn.Conv2d(self.MODEL_ASPP_OUTDIM+self.MODEL_SHORTCUT_DIM, self.MODEL_ASPP_OUTDIM, 3, 1, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(self.MODEL_ASPP_OUTDIM, self.MODEL_ASPP_OUTDIM, 3, 1, padding=1),
				nn.ReLU(inplace=True),
		)
		self.cls_conv = nn.Conv2d(self.MODEL_ASPP_OUTDIM, self.MODEL_NUM_CLASSES, 1, 1, padding=0)

		self.backbone = build_backbone(self.MODEL_BACKBONE, os=self.MODEL_OUTPUT_STRIDE)		
		self.backbone_layers = self.backbone.get_layers()

	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
#		feature_aspp = self.dropout(feature_aspp)
		feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat) 
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result

