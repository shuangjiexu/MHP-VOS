import os
import argparse
# require davis in https://github.com/fperazzi/davis-2017
import numpy as np
import cv2
from davis import cfg, io, DAVISLoader
from davis.misc.config import phase


def main(args):
    # Load dataset
    db = DAVISLoader(year=cfg.YEAR,phase=phase.TRAINVAL)
    anno_path = os.path.join(args.datapath, 'Annotations', '480p')
    # get trainval seq name
    with open(os.path.join(args.datapath, 'ImageSets', '2017', 'train.txt'), 'r') as train_f:
        train_list = [s.strip('\n') for s in train_f.readlines()]
    with open(os.path.join(args.datapath, 'ImageSets', '2017', 'val.txt'), 'r') as val_f:
        val_list = [s.strip('\n') for s in val_f.readlines()]
    trainval_list = train_list + val_list
    # scan all seq in trainval
    for class_name in trainval_list:
        print("solving %s ----------"%class_name)
        class_out_path = os.path.join(args.outpath, class_name)
        # save annotations for each obj
        for i in range(len(db[class_name].annotations)):
            # get obj path in a class
            annotation = db[class_name].annotations[i]
            for obj_id in np.unique(annotation)[1:]:
                tmp_path = os.path.join(class_out_path, str(obj_id-1))
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)
                obj_mask = (annotation==(obj_id)).astype('int')
                np.save(os.path.join(tmp_path, '%05d.png' % i), obj_mask)
        

if __name__ == '__main__':
    # Data settings
    parser = argparse.ArgumentParser(description='Prepare Data for DAVIS-2017')
    parser.add_argument('--datapath', default='/data1/shuangjiexu/data/DAVIS_2017', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--outpath', default='IndexedSegmentation', type=str,
                        help='name of experiment')
    args = parser.parse_args()
    args.outpath = os.path.join(args.datapath, args.outpath, "480p")

    main(args)
