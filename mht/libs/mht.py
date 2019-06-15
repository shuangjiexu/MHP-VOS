import os
import numpy as np
import math
import cv2

from treelib import Node, Tree
import networkx as nx

# from libs.kalman import KalmanFilter
from libs.kalman_filter import KalmanFilter
from libs.mwis import MWIS
from libs.motion_model import motion_model, generate_gaussian_map
from libs.warp_mask import warp_mask

from libs.deeplabv3plus import Deeplabv3plus
from libs.reID import ReidNetwork
from libs.osvos import OSVOS
from libs.osvos_deeplab import DeeplabOSVOS

import libs.utils as utils

import pickle
import matplotlib.pyplot as plt

class MHT():
    """Multiply Hypothesis tracking class
    """

    def __init__(self, config, dataLoader, sequence):
        """
        obj_id: start from 0, no bg
        """
        self.config = config
        self.dataLoader = dataLoader
        self.sequence = sequence
        # get object number in this sequence
        self.obj_num = utils.get_obj_num(os.path.join(self.config.mask_path,sequence,'00000.png'))
        # Trees store all tracks
        self.trackTrees = {}
        self.deeplab = []
        # self.osvos = []
        for obj_id in range(self.obj_num):
            self.trackTrees[obj_id] = []
            self.deeplab.append(Deeplabv3plus(sequence, obj_id+1, flow_dir=config.flow_dir, img_dir=config.img_dir))
            # self.osvos.append(DeeplabOSVOS(sequence, obj_id+1, flow_dir=config.flow_dir, img_dir=config.img_dir))
        # Nodes in the last frame (for gating)
        # { Tree Number:[node1,node2,...] }
        # self.currentNode = {}
        self.mwis = MWIS('track')
        # deeplab
        # self.reid = ReidNetwork('best')
        self.osvos_old = []
        for obj_id in range(self.obj_num):
            self.osvos_old.append(OSVOS(self.config, sequence, obj_id+1))
        # pre-process
        self.processData()
        # get all re-id score
        # if os.path.exists(os.path.join(self.config.debug_path, 're_id', '%s.score'%sequence)):
        #     print('loading re-id score from disk -------------------')
        #     with open(os.path.join(self.config.debug_path, 're_id', '%s.score'%sequence),'rb') as f:
        #         self.reid_scores = pickle.load(f)
        # else:
        #     if not os.path.exists(os.path.join(self.config.debug_path, 're_id')):
        #         os.makedirs(os.path.join(self.config.debug_path, 're_id'))
        #     with open(os.path.join(self.config.debug_path, 're_id', '%s.score'%sequence),'wb') as f:
        #         self.reid_scores = self.reidAll()
        #         pickle.dump(self.reid_scores, f)
        # print(self.reid_scores[6])
        self.test_arr = []

    # def reidAll(self):
    #     """ make a list of the re-id score matrix
    #     format: [ array(M*N) ] * T, T means frame number, M means number of detection, N means number of obj
    #     """
    #     data = self.dataLoader.content
    #     reid_scores = []
    #     path, img_list = utils.get_obj_img(self.config, self.sequence)
    #     for detections in range(len(data)):
    #         print('processing frame %d ---------------'%detections)
    #         rois = data[detections]['rois'] # (N,4)
    #         roi_num = rois.shape[0]
    #         scores = np.zeros((roi_num, len(img_list)))
    #         path_target = os.path.join(self.config.img_dir, self.sequence, '%05d.jpg'%(detections))
    #         for roiId in range(roi_num):
    #             roi = rois[roiId] # (y1,x1,y2,x2)
    #             bbox = [roi[1], roi[0], roi[3], roi[2]]
    #             for i in range(len(img_list)):
    #                 scores[roiId, i] = self.reid.compute_score(path, path_target, img_list[i], bbox)
    #         reid_scores.append(scores)
    #     return reid_scores


    def processData(self):
        # cut detections based on its confidence score
        self.dataLoader.cutWithScore(self.config.minDetScore)
        self.dataLoader.nms(self.config.ov_threshold)

    def iterTracking(self):
        # filter the data with detection score and overlap nms
        self.processData()
        data = self.dataLoader.content
        for i in range(len(data)): # len(data)
            print('current processing ------------------- '+str(i))
            # build and update track families
            self.formTrackFamily(data[i], i)
            # update the incompability list
            for treeId in range(len(self.trackTrees)):
                print('current object ------------------- '+str(treeId))
                # generate the global hypothesis
                paths, best_solution = self.treeToGraph(i+1, treeId)
                print('before pruning -------------------------------')
                for track in self.trackTrees[treeId]:
                    track.show()
                # N scan pruning
                self.nScanPruning(paths, best_solution, treeId)
                print('after pruning -------------------------------')
                for track in self.trackTrees[treeId]:
                    track.show()
                # save the output
                # tree.save2file('outs/tree/%d.txt'%i)
        # get best results
        results = []
        for obj_id in range(self.obj_num):
            results.append(self.findBestSolution(obj_id))
        return results


    def updateScore(self, scores, init=False):
        """Update score for each track
        inputs:
            Node: last node in tree (have previous score)
            appearance score (self.reid_scores)
            motion score (Mahalanobis distance and co-variance)
            scores = {'detection':scores[i], 'current_reid':current_reid, 'inverse_reid':inverse_reid,
                        'current_motion':current_motion, 'inverse_motion':inverse_motion, 'mask_iou':mask_iou}
        """
        # TODO: use mask iou to replace the distance score
        if init:
            # init score for root of a tree
            score = math.log(scores['detection'])
        else:
            # update by detection, distance and re-id score
            w_app = self.config.w_app
            w_mot = self.config.w_mot
            w_app_inv = self.config.w_app_inv
            w_mot_inv = self.config.w_mot_inv
            w_mask = self.config.w_mask
            # S_app = -1*math.log(0.5+0.5*math.exp(2.0*scores['current_reid'])) - math.log(self.config.c1)
            # S_app_inv = -1*math.log(0.5+0.5*math.exp(2.0*scores['inverse_reid'])) - math.log(self.config.c1)
            S_mot = scores['current_motion']
            S_mot_inv = scores['inverse_motion']
            S_mot_mask = scores['mask_iou']
            score = w_mot*S_mot + w_mot_inv*S_mot_inv + w_mask*S_mot_mask
            print('score -----------------------------------------')
            # print('S_app: %f'%S_app)
            # print('S_app_inv: %f'%S_app_inv)
            print('S_mot: %f'%S_mot)
            print('S_mot_inv: %f'%S_mot_inv)
            print('S_mot_mask: %f'%S_mot_mask)
            print('score: %f'%score)
            self.test_arr.append(score)
            print('-----------------------------------------------')
        return score


    def nScanPruning(self, paths, best_solution, treeNo, N=3, Bth=100):
        """Track Tree Pruning
        inputs:
            paths: list of track, {'treeId': treeId, 'track': [leaves to root], 'track_list': '0013424300'}
            best_solution: list of track number in path
            N: N-scan pruning approach
            Bth: branches number threshold
        """
        if paths == [] or best_solution == []:
            return
        T = len(paths[0]['track_list'])
        if T <= N:
            return
        # N Pruning
        ## Get k-(N-1) node in each tree, prun others
        for treeId in range(len(self.trackTrees[treeNo])):
            path_this_id = [paths[x] for x in best_solution if paths[x]['treeId']==treeId]
            node_in_path = []
            # find valid node in frame k-(N-1)
            for path in path_this_id:
                for track in path['track']:
                    if track.split('_')[1] == str(T-1-(N-1)):
                        node_in_path.append(track)
            # prune node if not in node_in_path
            node_in_tree = self.trackTrees[treeNo][treeId].filter_nodes(func=lambda x: x.identifier.split('_')[1] == str(T-1-(N-1)))
            node_names = [x.identifier for x in node_in_tree]
            for nodeId in node_names:
                # if nodeId.split('_')[1] == str(T-1-(N-1)):
                if nodeId not in node_in_path:
                    self.trackTrees[treeNo][treeId].remove_subtree(nodeId)
            '''
            # find K best and keep
            tracks = self.trackTrees[treeId].paths_to_leaves()
            if len(tracks) > Bth:
                # get weights and sort
                weights = [self.trackTrees[treeId].nodes[x[-1]].data['score'] for x in tracks]
                cut_index = 
            '''
        # remove empty tree
        self.trackTrees[treeNo] = [tree for tree in self.trackTrees[treeNo] if tree.nodes!={}]
        
    def findBestSolution(self, treeNo):
        data = self.dataLoader.content
        paths, results = self.treeToGraph(len(data), treeNo, timeConflict=True)
        path_result = [paths[x] for x in results]
        # combine result in time
        roi_numbers = []
        for i in range(len(data)):
            track_list = []
            for path in path_result:
                track_list.append(path['track_list'][i])
            none_zeros = [x for x in track_list if x != 0]
            # none zero roi number should be 1 or many with the same value 
            if len(none_zeros) == 0:
                roi_numbers.append(-1)
            elif len(none_zeros) == 1:
                roi_numbers.append(none_zeros[0]-1)
            elif len(none_zeros) > 1:
                # first check the result
                if none_zeros[1:] == none_zeros[:-1]:
                    roi_numbers.append(none_zeros[0]-1)
                else:
                    raise Exception('same value in track, check wmis')
        return roi_numbers

    def treeToGraph(self, T, treeNo, timeConflict=False):
        """chnage all the tree in self.trackTrees in a Graph for MWIS.
        inputs:
            T: total time in the target video
        """
        paths = []
        for treeId in range(len(self.trackTrees[treeNo])):
            # get all track path in a tree
            tracks = self.trackTrees[treeNo][treeId].paths_to_leaves()
            for track in tracks:
                # track: list of node identifier from root to leaf
                # change each track with identifier to number such as 0013430012
                # 0 represent the absent of the frame. detection start from 1
                # score updating
                track_list = [0]*T
                for node in track:
                    index = int(node.split('_')[3]) + 1  # index should from 1, 0 means missing
                    time = int(node.split('_')[1])
                    track_list[time] = index
                # score
                score = self.trackTrees[treeNo][treeId].nodes[track[-1]].data['score']
                paths.append({'treeId': treeId, 'obj_id':treeNo, 'track': track, 'track_list': track_list, 'weight': score})
        # judge whether there is an edge between two node
        def ifConnectedInRoi(node1, node2):
            for trackId in range(len(node1['track_list'])):
                if node1['track_list'][trackId] == 0 or node2['track_list'][trackId] == 0:
                    continue
                if node1['track_list'][trackId] == node2['track_list'][trackId]:
                    return True
            return False
        def ifConnectedInTime(node1, node2):
            for trackId in range(len(node1['track_list'])):
                if node1['track_list'][trackId] != 0 and node2['track_list'][trackId] != 0 and node1['track_list'][trackId] != node2['track_list'][trackId]:
                    return True
            return False
        ifConnected = ifConnectedInTime if timeConflict else ifConnectedInRoi
        # get all tracks, now get edges
        edges = []
        for i in range(len(paths)-1):
            for j in range(i+1, len(paths)):
                # judge two node
                if ifConnected(paths[i], paths[j]):
                    edges.append((i,j))
        # get weight for each track
        # TODO: for each track, we get a score
        weights = []
        for path in paths:
            print(path['weight'])
            weights.append(path['weight'])
        # write graph
        graph_dict = {}
        print(paths)
        print(edges)
        for edge in edges:
            if edge[0] in graph_dict:
                graph_dict[edge[0]].append(edge[1])
            else:
                graph_dict[edge[0]] = [edge[1]]
            if edge[1] in graph_dict:
                graph_dict[edge[1]].append(edge[0])
            else:
                graph_dict[edge[1]] = [edge[0]]
        # write graph file
        str_list = ['%d %d\n'%(len(paths), len(edges))]
        has_graph = False
        for i in range(len(paths)):
            if i in graph_dict:
                # notice that the edge of mwis is from 1
                str_list.append(' '.join(str(e+1) for e in graph_dict[i])+'\n')
                has_graph = True
            else:
                str_list.append('\n')
        if not has_graph:
            return [], []
        # write weight file
        str_weights = ['%d %f\n'%(i, weights[i]) for i in range(len(weights))]
        self.mwis.write_graph(str_list, str_weights)
        # find best solution
        results = self.mwis.local_search()
        assert results != -1
        print('best solation found--------------')
        print(results)
        return paths, results
        
            
    def generateGlobalHypothesis(self, T):
        pass

    def refineMask(self, frame, roi, mask, obj_id, expand_rate=None):
        """Use deeplab to get the refined mask
        frame: frame number
        roi: [y1,x1,y2,x2]
        """
        if not expand_rate:
            expand_rate = self.config.expand_rate
        img_path = os.path.join(self.config.img_dir, self.sequence, '%05d.jpg'%frame)
        # mask, _ = utils.load_mask(os.path.join(self.config.mask_path, self.sequence, '%05d.png'%frame), self.obj_id+1) 
        # bbox: [x1, y1, x2, y2], roi: [y1,x1,y2,x2]
        bbox = utils.bbox_from_roi(mask.shape, roi, self.config.expand_rate)
        # we fuse mask with osvos result
        # osvos_mask,_ = self.osvos.get_segmentation(frame)
        # mask_in = utils.fuse_mask(self.config, osvos_mask, mask)
        # M t-1 need to be flow to the next frame
        img_dir = os.path.join(self.config.img_dir, self.sequence)
        flow_dir = os.path.join(self.config.flow_dir, self.sequence)
        mask_in,_,_,_ = warp_mask(mask, frame-1, frame, flow_dir, img_dir)
        
        # get osvos
        # mask_osvos = self.osvos[obj_id].compute_mask(img_path, mask, bbox)
        mask_osvos,_ = self.osvos_old[obj_id].get_segmentation(frame)
        # mask_osvos = np.zeros(mask_in.shape)
        # fuse warped mask with osvos results
        mask_fuse = utils.fuse_mask(self.config, mask_in,mask_osvos)
        # mask_fuse = utils.merge_overlapped_blobs(self.config, mask_in,mask_osvos)
        result = self.deeplab[obj_id].compute_mask(img_path, mask_fuse, bbox)
        # remove small blob
        # result = utils.mask_remove_small_blobs(self.config, result)
        '''
        # get bbox from mask
        bbox_margined = utils.extract_bboxes(result>0.1)
        # if new bbox smaller than bbox in, we keep the lager one
        if (bbox_margined[2]-bbox_margined[0]+1)*(bbox_margined[3]-bbox_margined[1]+1) < (bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1):
            bbox_margined = bbox
        '''
        #cv2.rectangle(result, pt1=(bbox_margined[0],bbox_margined[1]),pt2=(bbox_margined[2],bbox_margined[3]), color=(1), thickness=2)
        #plt.imshow(result)
        #plt.show()
        return result, mask_in

    def formTrackFamily(self, detections, t):
        """build track tree in self.trackTrees
        inputs:
            detections: the detection results in one frame {'rois':[N,4], 'scores':[N], 'class_ids':[N]}
            t: time, int, from 0, 0 is the ground truth
        """
        rois = detections['rois']
        detections_scores = detections['scores']
        class_ids = detections['class_ids']
        img_path = os.path.join(self.config.img_dir, self.sequence, '%05d.jpg'%t)
        img_size = cv2.imread(img_path).shape[:2]
        if t == 0:
            # build from ground truth
            path = os.path.join(self.config.mask_path, self.sequence, '00000.png')
            for obj_id in range(self.obj_num):
                mask = utils.load_mask(path, obj_id+1)
                bbox = utils.extract_bboxes(mask>0.1) 
                # create a root node
                updated_score = self.config.initScore # self.updateScore(1.0, 0, 0, 0, init=True)
                tree = Tree()
                tree.create_node(tag="T_"+str(t)+"_N_"+str(obj_id), identifier="t_"+str(t)+"_n_"+str(obj_id), 
                                    data={'score':updated_score, 'mask':mask, 'bbox':bbox, 'history_bbox':[bbox]})
                self.trackTrees[obj_id].append(tree)
        else:
            # Gating for node section    
            # Using Kalman Filter or TCN prediction
            '''
            tempCurrentNode = {}
            for treeID in self.currentNode:
                # for each node in self.currentNode, do gating process and add nodes
                for node in self.currentNode[treeID]:
                    self.
            '''
            tempCurrentNode = {}
            for obj_id in range(self.obj_num):
                nodeObjs = []
                for treeId in range(len(self.trackTrees[obj_id])):
                    for node in self.trackTrees[obj_id][treeId].leaves():
                        if int(node.identifier.split('_')[1]) == (t - 1):
                            # this node is from t-1 frame
                            #x_bar, P_bar = node.data['kf'].predict()
                            # predict bbox using history
                            bbox_pred = motion_model(img_size, node.data['history_bbox'], t)
                            nodeObjs.append( {'treeId':treeId, 'node':node.identifier, 'obj_id': obj_id,
                                                'mask':node.data['mask'],'bbox_pred':bbox_pred, 'bbox':node.data['bbox']} )
                tempCurrentNode[obj_id] = nodeObjs
            print('roi number: %d'%rois.shape[0])
            # if the roi has no gate with any object
            for i in range(rois.shape[0]):
                print('current roi number is %d---------------------------------------------------'%i)
                roi_gating = False
                # for each detections, judge distance
                roi = rois[i] # (y1,x1,y2,x2)
                print(roi)
                bbox = [roi[1],roi[0],roi[3],roi[2]]

                ### new start
                roi_score = []
                for obj_id in range(self.obj_num):
                    obj_score = []
                    for nodeRecord in tempCurrentNode[obj_id]:
                        # compute for all of the score
                        
                        bbox_node = nodeRecord['bbox']
                        # compute scores
                        d = self.gating(bbox, bbox_node)
                        obj_score.append(d)
                    roi_score.append(obj_score)
                
                # we get roi bbox iou score with all nodes, now choose tree
                for obj_id in range(self.obj_num):
                    print('current obj is %d -----------------------'%obj_id)
                    # judge the app score of this roi
                    # current_reid = self.reid_scores[t][i,obj_id]
                    # print('re-id score is %f'%current_reid)
                    # if current_reid >= self.config.appScoreLimit:
                        # continue
                    # inverse_reid score is the min value of scores except current obj
                    # other_reids = [self.reid_scores[t][i,x] for x in range(len(self.reid_scores[t][i])) if x != obj_id]
                    # if other_reids == []:
                        # inverse_reid is None if there only one obj
                        # inverse_reid = 0
                    # else:
                        # inverse_reid = min(other_reids)
                    # get inverse motion score
                    other_motions = []
                    for j in range(len(roi_score)):
                        if j != obj_id:
                            other_motions.extend(roi_score[j])
                    if other_motions == []:
                        # inverse_reid is None if there only one obj
                        inverse_motion = 0
                    else:
                        inverse_motion = max(other_motions)
                    count = 0
                    for nodeId in range(len(tempCurrentNode[obj_id])):
                        print('current is roi %d, obj %d, node %s'%(i,obj_id,tempCurrentNode[obj_id][nodeId]['node']))
                        nodeRecord = tempCurrentNode[obj_id][nodeId]
                        # get current motion score
                        current_motion = roi_score[obj_id][nodeId]
                        # gating the node:
                        print('distance with last frame number %d is %f'\
                                %(int(nodeRecord['node'].split('_')[3]),current_motion))
                        if current_motion > self.config.dth:
                            # gating success 
                            # get mask result
                            mask_out,mask_in = self.refineMask(t, roi, nodeRecord['mask'], obj_id)
                            mask_out = utils.valid_mask(mask_out)
                            mask_iou = utils.calc_mask_iou(mask_out, nodeRecord['mask'])
                            # if no mask iou, skip
                            if mask_iou == 0:
                                continue
                            # get all the score we need for a new roi and target leaves
                            scores = {'detection':detections_scores[i],
                                        'current_motion':current_motion, 'inverse_motion':inverse_motion, 'mask_iou':mask_iou}
                            print(scores)
                            node_score = self.updateScore(scores)
                            current_score = self.trackTrees[obj_id][nodeRecord['treeId']]\
                                                            .nodes[nodeRecord['node']].data['score']
                            updated_score = node_score + current_score
                            # update history bbox
                            parent = self.trackTrees[obj_id][nodeRecord['treeId']]
                            history_bbox = self.trackTrees[obj_id][nodeRecord['treeId']]\
                                                    .nodes[nodeRecord['node']].data['history_bbox']
                            history_bbox.append(bbox)
                            # add node
                            print('creating node with %s'%(nodeRecord['node']))
                            self.trackTrees[obj_id][nodeRecord['treeId']]\
                                .create_node(tag="T_"+str(t)+"_N_"+str(i), 
                                    identifier="t_"+str(t)+"_n_"+str(i)+"_"+str(count), 
                                    parent=parent.nodes[nodeRecord['node']],
                                    data={'score':updated_score, 'bbox':bbox, 'mask':mask_out, 'history_bbox':history_bbox})
                            count = count + 1
                            roi_gating = True
                
                if not roi_gating:
                    # add a new tree to all tracks if the object not gate
                    for obj_id in range(self.obj_num):
                        # out of gating region
                        # add a new tree from the i-th detection
                        tree = Tree()
                        # create a root node
                        '''
                        ## use zero mask as input
                        mask_in = np.zeros(img_size)
                        mask, bbox = self.refineMask(t, roi, mask_in)
                        # if there is no mask, do not build a new track
                        if (bbox == np.array([0,0,0,0])).all():
                            print('no mask find')
                            continue
                        '''
                        ## score
                        # add OSVOS results for new mask
                        mask_out,mask_in = self.refineMask(t, roi, np.zeros(img_size), obj_id)
                        updated_score = self.updateScore({'detection':detections_scores[i]},init=True)
                        tree.create_node(tag="T_"+str(t)+"_N_"+str(i), identifier="t_"+str(t)+"_n_"+str(i), 
                                            data={'score':updated_score, 'bbox':bbox, 'mask':mask_out, 'history_bbox':[bbox]})
                        self.trackTrees[obj_id].append(tree)
                        


    def gating(self, curr_bbox, bbox):
        """Gating with predicted bbox
        curr_bbox: bbox of this roi
        bbox: [x1,y1,x2,y2] predicted
        """
        # use bbox iou? or mask iou
        bbox_iou = utils.calc_bbox_iou(curr_bbox, bbox)
        return bbox_iou

    '''
    def gating(self, y, x, P):
        """Gating with Kalman filter
        """
        mahalanobis_dist = (x-y).T @ np.linalg.inv(P) @ (x-y)
        mahalanobis_dist = mahalanobis_dist[0][0]
        motion_score = math.log(self.config.V / (math.pi*2)) - 0.5*mahalanobis_dist - 0.5*math.log(np.linalg.norm(P,ord=1) )
        return mahalanobis_dist, motion_score
    '''
        


    