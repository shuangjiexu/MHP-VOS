import numpy as np
from PIL import Image
import skimage.measure as measure
import cv2
import scipy.signal as signal

def get_obj_num(path):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    obj_num = len(np.unique(mask))
    return obj_num

def load_mask(path, obj_id):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    mask = mask.copy()
    mask[mask!=obj_id] = 0
    mask[mask!=0] = 1
    return mask.astype(np.float32), np.array(label.getpalette())

def imwrite_index(filename,array,color_palette):
    """ Save indexed png."""

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im =im.convert("P")
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

if __name__=='__main__':
    a = Image.open('./00000.png')
    print np.array(a)[300:400, 300:400]