import os
import copy

import numpy as np
import torch

from .vgg import *

def setup(opt):
    
    if opt.arch == 'vgg16':
        model = vgg16(pretrained=opt.pretrained)
    elif opt.caption_model == 'vgg16_bn':
        model = vgg16_bn(pretrained=opt.pretrained)
    # img is concatenated with word embedding at every time step as the input of lstm
    elif opt.caption_model == 'vgg19':
        model = vgg19(pretrained=opt.pretrained)
    # FC model in self-critical
    elif opt.caption_model == 'vgg19_bn':
        model = vgg19_bn(pretrained=opt.pretrained)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))
        
    return model