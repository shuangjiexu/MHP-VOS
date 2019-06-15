import numpy as np
import sys,os
import cv2

import torch

def valid_mask(cfgs, mask):
    # write origion mask
    mask_valid = mask.copy()
    mask_valid[mask_valid>cfgs.VALID_TH] = 1
    mask_valid[mask_valid<=cfgs.VALID_TH] = 0
    return mask_valid.astype(np.float32)

class OSVOS():
    """
    osvos v2 class for mask tracker
    """

    def __init__(self, cfgs, seqname, obj_id):
        """
        obj_id from 1
        """
        self.seqname = seqname
        self.obj_id = obj_id
        self.cfgs = cfgs
        self.results = None
        self.net = None
        if cfgs.osvos_path is not None:
            img_path = os.path.join(cfgs.img_dir, self.seqname)
            names_img = np.sort([file_[:-4] for file_ in os.listdir(img_path) if not os.path.isdir(os.path.join(img_path,file_))])
            # 1
            result_path_1 = os.path.join(cfgs.osvos_path, self.seqname, str(self.obj_id))
            img_list_1 = list(map(lambda x: os.path.join(result_path_1, "%05d"%int(x)+'_1osvos.png'), names_img))
            # add
            self.results = [img_list_1]
        else:
            raise Exception("Invalid OSVOS!")

    def get_segmentation(self, img_id):
        res_num = len(self.results)
        masks = []
        for i in range(res_num):
            if self.results[i][img_id] is not None:
                print(self.results[i][img_id])
                mask = cv2.imread(self.results[i][img_id])
                # print(self.results[i][img_id])
                mask = np.squeeze(mask)
                if len(mask.shape) == 3:
                    mask = mask[...,0]
                masks.append(mask/255.0)
            else:
                raise Exception("No OSVOS Results OR NET!")
        gt_mask = np.zeros(masks[0].shape).astype(np.float)
        for i in range(len(masks)):
            gt_mask = gt_mask + masks[i]
        gt_mask = gt_mask / res_num
        gt_mask[gt_mask<0.4] = 0
        return valid_mask(self.cfgs, gt_mask), gt_mask

            
        