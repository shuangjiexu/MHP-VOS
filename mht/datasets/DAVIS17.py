import os
import json
import numpy as np

import torch

from nms.nms_wrapper import nms

class DataLoader():
    """load detection data in json file
    """
    def __init__(self, json_path):
        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        self.content = self.solveData(data)

    def solveData(self, data):
        """convert data from json file to numpy array:
        inputs:
            data: from json file
        outputs:
            list of frames number, [ [ {'rois':[N,4], 'scores':[N], 'class_ids':[N]}, ... ], ... ]
        """
        content = []
        for key in sorted(data.keys()):
            detection_content = {}
            detection_content['rois'] = np.array(data[key]['rois'])
            detection_content['scores'] = np.array(data[key]['scores'])
            detection_content['class_ids'] = np.array(data[key]['class_ids'])
            content.append(detection_content)
        return content


    def cutWithScore(self, th):
        """select detections which have a higher score then threshold value
        inputs:
            th: threshold value
            self.content: list of frames number, [ {'rois':[N,4], 'scores':[N], 'class_ids':[N]}, ... ]
        outputs:
            same as self.content but without low scroes
        """
        for key_record in range(len(self.content)):
            temp_ind = np.arange(self.content[key_record]['scores'].shape[0])
            ind = temp_ind[self.content[key_record]['scores']>=th]
            self.content[key_record]['scores'] = self.content[key_record]['scores'][ind]
            self.content[key_record]['rois'] = self.content[key_record]['rois'][ind]
            self.content[key_record]['class_ids'] = self.content[key_record]['class_ids'][ind]



    def nms(self, nms_threshold):
        # Non-max suppression
        for key_record in range(len(self.content)):
            if self.content[key_record]['rois'] != np.array([]):
                keep = nms(torch.cat((torch.from_numpy(self.content[key_record]['rois']).float(), 
                            torch.from_numpy(self.content[key_record]['scores']).unsqueeze(1).float()), 1), nms_threshold)
                ind = keep.numpy()
                self.content[key_record]['scores'] = self.content[key_record]['scores'][ind]
                self.content[key_record]['rois'] = self.content[key_record]['rois'][ind]
                self.content[key_record]['class_ids'] = self.content[key_record]['class_ids'][ind]


def main():
    dataLoader = DataLoader('libs/test/aerobatics.json')
    print(dataLoader.content[0]['rois'].shape)
    print(dataLoader.content[0]['scores'].shape)
    print(dataLoader.content[0]['class_ids'].shape)
    dataLoader.cutWithScore(0.6)
    print(dataLoader.content[0]['rois'].shape)
    print(dataLoader.content[0]['scores'].shape)
    print(dataLoader.content[0]['class_ids'].shape)
    dataLoader.nms(0.6)
    print(dataLoader.content[0]['rois'].shape)
    print(dataLoader.content[0]['scores'].shape)
    print(dataLoader.content[0]['class_ids'].shape)
    #print(dataLoader.content[0])
    #print(len(dataLoader.content[0]))

if __name__ == '__main__':
    main()
                    