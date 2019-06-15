# MHP-VOS
Code for CVPR 19 paper: MHP-VOS: Multiple Hypotheses Propagation for Video Object Segmentation

### Introduction
we have 5 parts as following:
1. python2.7 + pycuda: [pylucid](https://github.com/yelantingfeng/pyLucid)
2. python2.7 + caffe: [flownet2](https://github.com/lmb-freiburg/flownet2)
3. python3.5 + pytorch0.3.1: [mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn)
4. python3.5 + pytorch0.4.1: [deeplabv3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
5. Python 3.6 + pytorch0.4.1: [mht](./mht)

1-4 in folder [libs](./libs)

5 in folder [mht](./mht)

For more details, please refer to the README.md file in each folder

### Inference 
To show the perfomance of our tracking tree, we give an example inference on the carousel(test-dev) video.
First of all, please download the [prepare](./prepare) file and unzip it under the current folder. [GoogleDrive](https://drive.google.com/open?id=1kHTmaNarpWftKoFktK7qazLyJX0ezaz2) [BaiDuYun](https://pan.baidu.com/s/1b4g6kaRlccQh7oLYT76-fw) (code:2xwv)

The structure of [prepare](./prepare) file looks like:

```
|--prepare

|----DAVIS_2017 #contains carousel video
|------Annotations
|------ImageSets
|------JPEGImages

|----deeplab_model #contains the [deeplabv3+](https://github.com/jfzhang95/pytorch-deeplab-xception) models of each carousel object
|------carousel_1_99.pth
|------carousel_2_99.pth
|------carousel_3_99.pth
|------carousel_4_99.pth

|----mask_rcnn_result #contains the bbox proposals generated from the [mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn)
|------carousel.json

|----osvos_result #contains the segmentation results using osvos model
|------carousel

|----test_flow #contains optical flows generated with [flownet2](https://github.com/lmb-freiburg/flownet2)
|------carousel
```

Run the following code:

    $ cd mht
    $ python test_mht.py

and it will generate three folders (vis_detections, outs and final_results)
```
|--mht
|----vis_detections #detection bbox
|----outs #build tree for each objects
|----final results #tracking results
```

Or you can just run the main.py to see all the results including the final masks in $out$ file.

    $ cd mht
    $ python main.py

### Citation
If you use this code please cite:

```
@inproceedings{xu2019mhp,
  	title={MHP-VOS: Multiple Hypotheses Propagation for Video Object Segmentation},
  	author={Xu, Shuangjie and Liu, Daizong and Bao, Linchao and Liu, Wei and Zhou, Pan},
  	booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    year={2019}
}
```
