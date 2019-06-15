import numpy as np 
import matplotlib.pyplot as plt 

from torchvision import transforms

import skimage.measure as measure

from PIL import Image

def clip_around_bbox(bbox_in, imgsize):
    """ 
    bbox is in [[x1, y1], [x2, y2]]
    imgsize is in [h, w]
    """
    bbox_around = np.around(bbox_in)
    bbox = np.zeros((4))
    bbox[0] = max(0, bbox_around[0])
    bbox[1] = max(0, bbox_around[1])
    bbox[2] = min(imgsize[1]-1, bbox_around[2])
    bbox[3] = min(imgsize[0]-1, bbox_around[3])
    return bbox.astype(np.int16)

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width]. Mask pixels are either 1 or 0.

    Returns: bbox array [x1, y1, x2, y2].
    """
    # Bounding box.
    regions = measure.regionprops(mask)
    if len(regions):
        y1,x1,y2,x2 = regions[0].bbox 
        bbox = np.array([x1, y1, x2, y2])
        return bbox.astype(np.int32)
    else:
        return np.array([0,0,0,0])

def compute_bbox_from_mask(mask):
    # return [x1, y1, x2, y2]
    margin = 0.15
    h,w = mask.shape[:2]
    # [x1, y1, x2, y2]
    bbox = extract_bboxes((mask>0.1).astype(np.int16))
    h_box = bbox[3] - bbox[1]
    w_box = bbox[2] - bbox[0]
    # apply margin
    bbox_margined = [bbox[0]-margin*w_box, bbox[1]-margin*h_box, bbox[2]+margin*w_box, bbox[3]+margin*h_box]
    bbox_margined = clip_around_bbox(bbox_margined, mask.shape[:2])
    return bbox_margined

def read_flow(flowfile):
    f = open(flowfile, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)

mask = Image.open('./00000.png')

mask = np.array(mask, dtype=np.float32)

# print(mask.shape)

bbox = compute_bbox_from_mask(mask)

img = Image.open('./00000.jpg')
img1 = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(img)
im1 = transforms.ToTensor()
img1 = np.array(img1, dtype=np.float32)
img = np.array(img, dtype=np.float32)
plt.figure()
plt.title('display')
plt.subplot(211)
plt.imshow(img)
plt.subplot(212)
plt.imshow(img1)
plt.show()
# print(mask.shape)

flow = read_flow('00001.flo')

img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
mask = mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
flow = flow[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
# print(bbox)
# plt.imshow(flow[:,:,0])
# plt.show()
# flow = read_flow('00001.flo')
# print(flow.max())
# print(flow.min())
# plt.imshow(flow[:,:,1])
# plt.show()