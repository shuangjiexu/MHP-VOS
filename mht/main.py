import os
import numpy as np
import cv2
from PIL import Image
import pickle

from config.config import Config
from datasets.DAVIS17 import DataLoader
from libs.mht import MHT
from libs.utils import get_obj_num, load_mask, compute_bbox_from_mask, fuse_mask, imwrite_index
from libs.save_utils import *
from libs.merge_masks import merge_masks
from libs.warp_mask import warp_mask

class DavisConfig(Config):
    """Configuration for training on dataset.
    Derives from the base Config class and overrides values specific
    to the target task.
    """
    # Give the configuration a recognizable name
    TARGET_DATASET = 'test-dev'
    SAVE_PATH = 'out'
    DATA_PATH = 'data'
    json_path = '../prepare/mask_rcnn_result'
    minDetScore = 0.05
    ov_threshold = 0.60

# sequences = [ 'man-bike', 'girl-dog', 'cats-car', 'planes-crossing', 'slackline', 'mtb-race', 'subway']
# 'aerobatics', 'carousel', 'lock', 'gym', 'giant-slalom', 'golf',  'monkeys-trees', 'seasnake', 'salsa',
# 'chamaleon', 'guitar-violin', 'deer', 'tandem', 'tennis-vest', 'skate-jump', 'tractor', 
# 'car-race', 'orchid', 'people-sunset', 'hoverboard', 'rollercoaster', 'helicopter', 'horsejump-stick', 
#sequences = ['blackswan','bmx-trees','breakdance','camel','car-roundabout','car-shadow','cows','dance-twirl',
#            'dog','drift-chicane','drift-straight','goat','horsejump-high','kite-surf','libby','motocross-jump',
#            'paragliding-launch','parkour','scooter-black','soapbox']
# sequences = [ 'boat', 'dog', 'horse', 'motorbike', 'train']
# 'aeroplane', 'bird', 'car', 'cat', 'cow',  
sequences = ['carousel']
def main():
    config = DavisConfig()
    config.display()
    for sequence in sequences:  # os.listdir(config.img_dir)
        results_path = os.path.join(config.SAVE_PATH, 'final_results', sequence)
        # write first gt map
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        if os.path.isdir(os.path.join(config.img_dir, sequence)):
            # get obj_num
            obj_num = get_obj_num(os.path.join(config.mask_path, sequence, '00000.png'))
            # load json data
            dataLoader = DataLoader(os.path.join(config.json_path, '%s.json'%sequence))
            data = dataLoader.content
            
            # run MHT
            mht = MHT(config, dataLoader, sequence)

            # plot all detection for debug
            # all_reid_score = mht.reid_scores
            for obj_id in range(obj_num):
                for i in range(len(dataLoader.content)):
                    content_roi = dataLoader.content[i]['rois']
                    # current_reid = all_reid_score[i]
                    plot_detections(content_roi, i, obj_id, os.path.join(config.img_dir, sequence),
                                        targetPath=os.path.join(config.SAVE_PATH,'detections',sequence))
            
            # get bbox_list, [roi_number]*sequence number, -1 means no bbox
            roi_numbers = mht.iterTracking()

            ## save each path result
            for obj_id in range(len(mht.trackTrees)):
                for trackId in range(len(mht.trackTrees[obj_id])):
                    # track.show()
                    plot_tracks(mht.trackTrees[obj_id][trackId], trackId, obj_id, os.path.join(config.img_dir, sequence), 
                                    targetPath=os.path.join(config.SAVE_PATH,'tracks',sequence))
            plot_final_bbox(config, roi_numbers, data, sequence, 
                    targetPath=os.path.join(config.SAVE_PATH,'final_bbox',sequence))
            
            # get mask in these bbox
            masks = []
            final_result = []
            img_dir = os.path.join(config.img_dir, sequence)
            flow_dir = os.path.join(config.flow_dir, sequence)
            # add ground truth mask
            label = Image.open(os.path.join(config.mask_path, sequence, '00000.png'))
            mask = np.atleast_3d(label)[...,0]
            final_result.append(mask)
            imwrite_index(os.path.join(results_path, '00000.png'),mask,config.PALETTE)
            temp_mask = []
            for obj_id in range(obj_num):
                mask = load_mask(os.path.join(config.mask_path, sequence, '00000.png'), obj_id)
                temp_mask.append(mask)
            masks.append(temp_mask)
            bbox_all = []
            # circulation for merge mask
            for i in range(1,len(roi_numbers[0])):
                temp_mask = []
                temp_bbox = []
                for obj_id in range(obj_num):
                    last_mask = masks[-1][obj_id]
                    if roi_numbers[obj_id][i] == -1:
                        # do not have mask, we warp to get a predicted mask
                         # refine missing bbox if there is missed
                        mask_in,_,_,_ = warp_mask(last_mask, i-1, i, flow_dir, img_dir)
                        bbox = compute_bbox_from_mask(mask_in)
                        if bbox[3]-bbox[1] == 0 or bbox[2]-bbox[0] == 0:
                            roi = None
                            temp_bbox.append([0,0,2,2])
                        else:
                            roi = [bbox[1],bbox[0],bbox[3],bbox[2]]
                            temp_bbox.append(bbox)
                    else:
                        roi = data[i]['rois'][roi_numbers[obj_id][i]]
                        temp_bbox.append(data[i]['rois'][roi_numbers[obj_id][i]])
                    # mask propagation to get fine mask
                    if roi is not None:
                        mask_out, _ = mht.refineMask(i, roi, last_mask, obj_id, expand_rate=0.2)
                        temp_mask.append(mask_out)
                    else:
                        temp_mask.append(np.zeros(last_mask.shape))
                bbox_all.append(temp_bbox)
                masks.append(temp_mask)
                # get merged mask
                final_mask = merge_masks(config, masks[-1], final_result[-1], bbox_all[-1], i, sequence) 
                final_result.append(final_mask)
                # write final mask
                imwrite_index(os.path.join(results_path, '%05d.png'%i),final_mask,config.PALETTE)
            with open(os.path.join(results_path, 'bbox.sol'), 'wb') as f:
                pickle.dump(bbox_all, f)

if __name__ == '__main__':
    main()