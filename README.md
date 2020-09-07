# Yolo-security-camera

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

Security camera with YOLO object detection.

work with PaspberryPi.

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

Based by [keras-yolo3](https://github.com/qqwweee/keras-yolo3)

## Install

```
sudo apt-get install libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev

pip3 install numpy h5py pillow
pip3 install matplotlib
pip3 install keras==2.1.5
pip3 install tensorflow==1.14
export LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1
pip3 install opencv-python 
```

## Run

```
wget https://pjreddie.com/media/files/yolov3-tiny.weights
python3 convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolo-tiny.h5
```

#### Run
`python3 yolo_video.py --input /dev/video0 --output record/`

#### Stop
`touch record/CAM_STOP`

#### Simple script

* run: `securityCam.sh true`

* stop: `securityCam.sh false`
