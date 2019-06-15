from libs.mht import MHT
from datasets.DAVIS17 import DataLoader
from config.config import Config

import os
import cv2
import numpy as np
import pickle

class DavisConfig(Config):
    """Configuration for training on dataset.
    Derives from the base Config class and overrides values specific
    to the target task.
    """
    # Give the configuration a recognizable name
    # Vc
    Vc = 1920*1080
    pFalseAlarm = 1/(Vc)
    # detection TH
    minDetScore = 0.05
    ov_threshold = 0.60


def plot(img_path, data):
    # plot for test
    img = cv2.imread(img_path)
    for roi in data:
        cv2.rectangle(img, pt1=(roi[1],roi[0]),pt2=(roi[3],roi[2]), color=(0,255,0), thickness=2)
    cv2.imshow('image', img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def plot_tracks(tree, treeId, obj_id, imagePath, targetPath='outs'):
    paths = tree.paths_to_leaves()
    for pathId in range(len(paths)):
        bboxs = []
        for node in paths[pathId]:
            # save image 
            t = int(node.split('_')[1])
            bbox = tree.nodes[node].data['bbox'] # (y1,x1,y2,x2)
            bboxs.append(bbox)
            mask = tree.nodes[node].data['mask']
            img = cv2.imread(os.path.join(imagePath, '%05d.jpg'%t))
            cv2.rectangle(img, pt1=(bbox[0],bbox[1]),pt2=(bbox[2],bbox[3]), color=(0,255,0), thickness=2)
            img = cv2.putText(img, tree.nodes[node].identifier, (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
            # draw mask
            img[...,2]=np.where(mask>=0.1, 255, img[...,2])
            if not os.path.exists(os.path.join(targetPath, str(obj_id), str(treeId), str(pathId))):
                os.makedirs(os.path.join(targetPath, str(obj_id), str(treeId), str(pathId)))
            cv2.imwrite(os.path.join(targetPath, str(obj_id), str(treeId), str(pathId), '%05d.jpg'%t), img)
            with open(os.path.join(targetPath, str(obj_id), str(treeId), str(pathId), 'bbox.txt'), 'w') as f:
                f.write(str(bboxs))

def plot_detections(rois, id, obj_id, imagePath, targetPath='vis_detections'):
    img = cv2.imread(os.path.join(imagePath, '%05d.jpg'%id))
    for i in range(rois.shape[0]):
        roi = rois[i]
        # re_id_score = current_reid[i, obj_id]
        cv2.rectangle(img, pt1=(roi[1],roi[0]),pt2=(roi[3],roi[2]), color=(0,255,0), thickness=2)
        img = cv2.putText(img, '%d: %f'%(i, 0.), (roi[1],roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
        if not os.path.exists(targetPath):
            os.makedirs(targetPath)
        cv2.imwrite(os.path.join(targetPath, '%05d.jpg'%id), img)

def main():
    config = DavisConfig()
    config.display()

    seq_root = '../prepare/DAVIS_2017/JPEGImages/480p/carousel/'

    dataLoader = DataLoader('../prepare/mask_rcnn_result/carousel.json')
    obj_id = 0
    mht = MHT(config, dataLoader, 'carousel')

    #print(dataLoader.content[0]['rois'].shape)
    #mht.processData()
    # all_reid_score = mht.reid_scores
    # plot processed data
    for i in range(len(dataLoader.content)):
        content_roi = dataLoader.content[i]['rois']
        # current_reid = all_reid_score[i]
        plot_detections(content_roi, i, obj_id, seq_root)
    
    print(len(dataLoader.content))
    print(dataLoader.content[0]['rois'].shape)

    
    #for T in range(len(dataLoader.content)):
    #    plot('libs/test/carousel/%05d.jpg'%T, dataLoader.content[T]['rois'])
    roi_numbers = mht.iterTracking()
    for obj_id in range(len(mht.trackTrees)):
        for trackId in range(len(mht.trackTrees[obj_id])):
            # track.show()
            plot_tracks(mht.trackTrees[obj_id][trackId], trackId, obj_id, seq_root)
            '''
            with open('inputs/re_id/carousel/%d.tree'%trackId, 'wb') as f:
                pickle.dump(mht.trackTrees[trackId], f)
            '''
    # print(mht.trackTrees)
    # plot final results
    data = dataLoader.content
    targetPath = 'final_results'

    print(roi_numbers)
    
    for i in range(1,len(roi_numbers[0])):
        img_path = os.path.join(seq_root, '%05d.jpg'%i)
        img = cv2.imread(img_path)
        data_t = data[i]['rois']
        for obj_id in range(len(roi_numbers)):
            if roi_numbers[obj_id][i] == -1:
                continue
            roi = data_t[roi_numbers[obj_id][i]]
            cv2.rectangle(img, pt1=(roi[1],roi[0]),pt2=(roi[3],roi[2]), color=(0,255,0), thickness=2)
            # plot text
            img = cv2.putText(img, str(obj_id), (roi[1],roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
        if not os.path.exists(targetPath):
            os.makedirs(targetPath)
        cv2.imwrite(os.path.join(targetPath, '%05d.jpg'%i), img)


if __name__ == '__main__':
    main()
