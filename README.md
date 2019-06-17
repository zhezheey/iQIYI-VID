# Celebrity Video Identification Based on Face Features

### Introduction

This repository contains codes for [2019 iQIYI Celebrity Video Identification Challenge](http://challenge.ai.iqiyi.com/detail?raceId=5c767dc41a6fa0ccf53922e7), which achieved a mAP score of 0.8949 on the test set (**Ranked 6th**), inspired by [Jasonbaby](https://github.com/Jasonbaby) and created by Wenzhe Wang.

### Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Training](#training)
4. [Submission](#submission)
5. [Reference](#reference)

### Requirements

1. Python 3.5
2. tensorflow-gpu (I use 1.4.0)
3. Keras (I use 2.0.8)

### Installation

1. Clone the iQIYI-VID repository into `$VID_ROOT`
	```Shell
	git clone https://github.com/zhezheey/iQIYI-VID.git
	```

2. Install python packages you might not have in `requirements.txt`
	```Shell
	pip install -r requirements.txt
	```

### Training

1. Download the [IQIYI-VID](http://challenge.ai.iqiyi.com/detail?raceId=5c767dc41a6fa0ccf53922e7) dataset, then place `face_train_v2.pickle` and `face_val_v2.pickle` inside the `$VID_ROOT/feat` directory, `train_gt.txt` and `val_gt.txt` inside the `$VID_ROOT/data` directory.

2. Train the MLP models (see more details [here](train/README.md))
	```Shell
	cd $VID_ROOT/train
	python get_gt.py
	# Change the batch_size in train.py according to your GPU memory.
	sh train.sh
	```

3. By default, trained models are saved under `$VID_ROOT/train/model`.

### Submission

Follow the steps below to build the Docker image of our submission (see more details [here](docker/resources/README.md)).

1. Move the trained models into the `$VID_ROOT/docker/resources` directory.

2. Build the Docker image
	```Shell
	cd $VID_ROOT/docker
	docker build -t zheey:1.0 -f Dockerfile .
	```

### Reference

```
@article{liu2018iqiyi,
  title={iqiyi-vid: A large dataset for multi-modal person identification},
  author={Liu, Yuanliu and Shi, Peipei and Peng, Bo and Yan, He and Zhou, Yong and Han, Bing and Zheng, Yi and Lin, Chao and Jiang, Jianbin and Fan, Yin and others},
  journal={arXiv preprint arXiv:1811.07548},
  year={2018}
}
```
