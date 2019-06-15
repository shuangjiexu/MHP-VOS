import os
import cv2
import numpy as np

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

def plot_final_bbox(config, roi_numbers, data, sequence, targetPath='final_results'):
    for i in range(1,len(roi_numbers[0])):
        img_path = os.path.join(config.img_dir, sequence, '%05d.jpg'%i)
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