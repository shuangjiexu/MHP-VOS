import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize
from PIL import Image
import torch
import json

def get_obj_num(path):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    obj_num = len(np.unique(mask))
    return obj_num

with open('../data/DAVIS/test_dev/ImageSets/2017/test-dev.txt') as f:
    fr = f.readlines()
    file_namelist = [x.strip() for x in fr]
# while True:
#     if file_namelist[0] != 'carousel':
#         file_namelist.pop(0)
#     else:
#         break
for TARGET_CLASS in file_namelist:
    print('do with '+ TARGET_CLASS)

    # TARGET_CLASS = 'planes-crossing'

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Path to trained weights file
    # Download this file and place in the root of your
    # project (See README file for details)
    # MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")
    MODEL_PATH = os.path.join('./checkpoints/'+TARGET_CLASS+'.pth')

    # Directory of images to run detection on
    # IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    IMAGE_DIR = os.path.join('../data/DAVIS/test_dev/JPEGImages/480p', TARGET_CLASS)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    '''
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
    '''

    ############################################################
    #  Dataset
    ############################################################

    # from davis import Annotation

    # an = Annotation(TARGET_CLASS, single_object=0)
    OBJ_NUMBER = get_obj_num(os.path.join('../data/DAVIS/test_dev/Annotations/480p', TARGET_CLASS,'00000.png')) - 1 
    #OBJ_NUMBER = 1
    class_names = ['BG'] + ['obj_'+str(x) for x in range(1,OBJ_NUMBER+1)]
    print(class_names)

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        # GPU_COUNT = 0 for CPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + OBJ_NUMBER
        DETECTION_MIN_CONFIDENCE = 0.05
        DETECTION_NMS_THRESHOLD = 0.66

    config = InferenceConfig()
    config.display()

    # Create model object.
    model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights trained on MS-COCO
    model.load_state_dict(torch.load(MODEL_PATH))

    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]

    dir_results = os.path.join(ROOT_DIR, 'out/', TARGET_CLASS)
    if not os.path.isdir(dir_results):
        os.makedirs(dir_results)

    if not os.path.exists('./out/json'):
        os.mkdir('./out/json')
    #with open('./out/oneobject_json/'+TARGET_CLASS+'.json',"w") as f:
    total = {}
    for file_name in file_names:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

        # Run detection
        results = model.detect([image])
        # print(results[0])

        # Visualize results
        r = results[0]
        ax = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
        #plt.show()
        #skimage.io.imsave('test_out.jpg', masked_img)
        fig = ax.get_figure()
        fig.savefig(os.path.join(dir_results, file_name+'.png'))
        #ax.close()
        #r.pop('masks')
        r['rois'] = r['rois'].tolist()
        print(r['masks'])
        bb
        r['masks'] = r['masks'].tolist()
        r['class_ids'] = r['class_ids'].tolist()
        r['scores'] = r['scores'].tolist()
        total[file_name] = r
        #print(total)
    with open('./out/json_ldz/'+TARGET_CLASS+'.json',"w") as f:
        json.dump(total,f)
    #for file_name in file_names:
    #    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

    #    # Run detection
    #    results = model.detect([image])
    #    # print(results[0])

    #    # Visualize results
    #    r = results[0]
    #    ax = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                                class_names, r['scores'])
    #    plt.show()
    #    #skimage.io.imsave('test_out.jpg', masked_img)
    #    fig = ax.get_figure()
    #    fig.savefig(os.path.join(dir_results, file_name+'.png'))
    #    #ax.close()
    print('done!')
