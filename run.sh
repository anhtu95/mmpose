#!/bin/bash

python3 ./demo/top_down_video_demo_with_mmdet.py ./demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth --video-path $1 --out-video-root ./output/ --device $2

python3 ./drawing_projection.py