import os
import numpy as np

class Config:
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Path Param --------------------------------------------------------------
    dataset = 'DAVIS_2017'
    # img_path = 'input'
    flow_dir='../prepare/test_flow'
    img_dir='../prepare/DAVIS_2017/JPEGImages/480p'
    mask_path ='../prepare/DAVIS_2017/Annotations/480p'
    osvos_path = '../prepare/osvos_result'
    debug_path = 'inputs'
    PALETTE = np.loadtxt('config/palette.txt',dtype=np.uint8).reshape(-1,3)

    # Kalman Param ------------------------------------------------------------
    stateSize = 4
    observationSize = 2
    F = np.array([[1,0,1,0],
                  [0,1,0,1],
                  [0,0,1,0],
                  [0,0,0,1]])
    H = np.array([[1,0,0,0],
                  [0,1,0,0]])

    # Other Param -------------------------------------------------------------
    w_app = 0
    w_mot = 0.3
    w_app_inv = 0
    w_mot_inv = -0.3
    w_mask = 0.7
    c1 = 0.3
    c2 = 0.3

    initScore = 5

    V = 854*480
    # used when compute overlap between two mask
    DILATION_COEFFICIENT = 1.05
    expand_rate = 0
    dth = 0.2
    MIN_PIXEL_NUMBER = 20*20
    VALID_TH = 0.3
    OVERLAP_TH = 0.5
    appScoreLimit = 1.5

    ## MHT parameters
    # P_D
    pDetection = 0.90
    # measurement likelihood under the null hypthesis
    # Please refer to adjustOtherParameters.m or the paper to see how to set this parameter for different videos.      
    pFalseAlarm = 0.000001
    # B_{th}
    maxActiveTrackPerTree = 100
    # N_{miss}
    dummyNumberTH = 15
    # N (N scan)
    N = 5
    # d_{th}
    # set this parameter to 6 for motion-based tracking and 12 for motion+appearance-based tracking.                    
    MahalanobisDist = 12   

    ## appearance parameters
    # (1)'cnn': CNN feature  (2)'': No appearnace 
    appSel = 'cnn'
    # w_{app}
    # When appearance is not used (i.e. other_param.appSel = ''), the appearance parameters are ignored.
    appW = 0.9
    # w_{mot} = 1-w_{app}
    motW = 0.1
    # c2
    appTH = -0.8
    # c1                                              
    appNullprob = 0.3

    ## additional parameters
    # set this parameter to 0 for 2D tracking (e.g. MOT) and 1 for 3D tracking (e.g. PETS)
    is3Dtracking = 0
    # detection pruning. Detections whose confidence score is lower than this threshold are ignored.
    minDetScore = 0
    # nms overlap threshold
    ov_threshold = 0.6
    # confirmed track pruning (MOT). 
    # Confirmed tracks whose average detection confidence score is lower than this threshold are ignored.
    confscTH = 5
    # confirmed track pruning (PETS)
    # other_param.confscTH = 0.2
    # confirmed track pruning based on a ratio of the # of dummy observations to the # of total observations
    dummyRatioTH = 0.5
    # confirmed track pruning based on a track length
    minLegnthTH = 5
    # allowed bounding box scale difference between consecutive frames in each track. 
    # Set this parameter to > 1. For example, 1.4 means 40% scale change is allowed.  
    maxScaleDiff = 1.4
    

    def __init__(self):
        """Set values of computed attributes."""
        pass



    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
