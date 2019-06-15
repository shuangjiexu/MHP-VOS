# Code-for-CVPR2019
This is the basic code (except tree building) for our CVPR2019 paper.

### Introduction
we have 4 parts as following:
1. python2.7 + pycuda: [pylucid](https://github.com/yelantingfeng/pyLucid)
2. python2.7 + caffe: [flownet2](https://github.com/lmb-freiburg/flownet2)
3. python3.5 + pytorch0.3.1: [mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn)
4. python3.5 + pytorch0.4.1: [deeplabv3+](https://github.com/jfzhang95/pytorch-deeplab-xception)

### Get start
At first, please download DAVIS dataset (named DAVIS) under the ./data folder
1. lucid data generation

  first generate 200 multiple objects lucid images 

    $ cd pylucid
    $ python generate_val.py

  then seperate each mask into different files
  
    $ python lucid_mask.py
    
2. optical flow generation

  git clone the original respository in [flownet2](https://github.com/lmb-freiburg/flownet2) and follow its implements, and then generate optical flow list
  
    $ cd flownet2
    $ python generate_flow_list.py
   
  after getting the lucidpair.txt file, run the caffe to generate opticalflow
  
    $ python run-flownet-many.py ../models/flownet2-models/FlowNet2/FlowNet2_weights.caffemodel.h5 ../models/flownet2-models/FlowNet2/FlowNet2_deploy.prototxt.template ./lucidpair.txt

3. warp the mask with opticalflow

  warp the optical flow

    $ python compute_warpmask.py

4. proposal generation

  first train the pretrained (on ImageDataset) model on COCO dataset
  
    $ cd pytorch-mask-rcnn
    $ python coco.py
  
  next train on Davis2017 trainval set
  
    $ python davis2017.py
  
  then finetune on each video with lucid data
    
    $ python singlefile.py --dataset=/your/path --year=2017 --model=./logs/coco20180927T1009/mask_rcnn_coco_0160.pth --logs=./logs --limit=10 --augment method=xu_val_augment_2500 train
   
  at last, test on the sequences to generate coarse proposals
  
    $ python demo_multi.py

5. MHP 
  
  follow to ../mht

6. mask segmentation

  with the bounding boxes of objects in each frame and warpped previous mask, we train the pretrained (on COCO) deeplabv3+ on Davis2017 trainval set
  
    $ cd pytorch-deeplab-xception
    $ python train_offline.py
   
  then finetune on each video with corresponding lucid images
  
    $ python lucid_online.py
  
  and we can test on each video sequences
  
    $ python deeplabv3plus.py
  
  or
  
    $ python lucid_test.py
  
