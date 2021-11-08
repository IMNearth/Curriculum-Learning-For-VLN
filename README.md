# Curriculum Learning For LVN
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This is the PyTorch implementation of our paper:<br>
**Curriculum Learning for Vision-and-Language Navigation**<br>
*Jiwen Zhang, [Zhongyu Wei](http://www.sdspeople.fudan.edu.cn/zywei/), [Jianqing Fan](https://fan.princeton.edu/), [Jiajie Peng](https://jiajiepeng.github.io/)*<br>
35th Conference on Neural Information Processing Systems (NeurIPS 2021)



## Most Recent Events

* 2021-11-06: We uploaded the CLR2R dataset mentioned in our paper. (<code>tasks/R2R-judy/data</code>) 
* 2021-11-08: We update the installation instructions.





## Model architectures

This repository includes several SOTA navigation agents previously released. They are

* [Follower agent](https://github.com/ronghanghu/speaker_follower) (from University of California, Berkeley, Carnegie Mellon University and Boston University) released with paper [Speaker-Follower Models for Vision-and-Language Navigation](https://arxiv.org/pdf/1806.02724.pdf), by Fried, Daniel, Ronghang Hu, Volkan Cirik, Anna Rohrbach, Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko, Dan Klein and Trevor Darrell. *NeurIPS(2018)*.
* [Self-Monitoring agent](https://github.com/chihyaoma/selfmonitoring-agent) (from Georgia Institute of Technology, University of Maryland and Salesforce Research) released with paper [Self-Monitoring Navigation Agent via Auxiliary Progress Estimation](https://arxiv.org/pdf/1901.03035.pdf), by Ma, Chih-Yao, Jiasen Lu, Zuxuan Wu, Ghassan Al-Regib, Zsolt Kira, Richard Socher and Caiming Xiong. *ICLR(2019)*.
* [EnvDrop agent](https://github.com/airsplay/R2R-EnvDrop) (from UNC Chapel Hill) released with paper [Learning to Navigate Unseen Environments: Back Translation with Environmental Dropout](https://arxiv.org/pdf/1904.04195.pdf), by Tan, Hao, Licheng Yu and Mohit Bansal. *NAACL(2019)*.

and a path-instruction scorer

* [VLN-BERT](https://github.com/arjunmajum/vln-bert) (from Georgia Institute of Technology, Facebook AI Research and Oregon State University) released with paper [Improving Vision-and-Language Navigation with Image-Text Pairs from the Web](https://arxiv.org/pdf/2004.14973.pdf), by Majumdar, Arjun, Ayush Shrivastava, Stefan Lee, Peter Anderson, Devi Parikh and Dhruv Batra. *ECCV(2020)*.



## Installation

### Setting up Environments

1. Install Python 3.6 (Anaconda recommended: https://docs.anaconda.com/anaconda/install/index.html).

2. Install PyTorch following the instructions on https://pytorch.org/ (in our experiments, it isPyTorch 1.5.1+cu101).

3. Following build instructions in [this github](https://github.com/peteanderson80/Matterport3DSimulator/tree/v0.1) to build up a v0.1 Matterport3D simulator.

   Besides, just in case you have an error when compiling the simulator, you can try this

   ```bash
   mkdir build && cd build
   cmake -D CUDA_TOOLKIT_ROOT_DIR=path/to/yout/cuda
   make
   cd ../
   ```

   For more details on the Matterport3D Simulator, you can refer to [`README_Matterport3DSimulator.md`](https://github.com/ronghanghu/speaker_follower/blob/master/README_Matterport3DSimulator.md).



### Dataset Download

Luckily, this repository contains the R2R dataset and CLR2R dataset, so you ONLY have to download precomputing ResNet image features from [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator). 

- [ResNet-152-imagenet features [380K/2.9GB]](https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip?dl=1)
- [ResNet-152-places365 features [380K/2.9GB]](https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip?dl=1)

Download and extract the tsv files into the `img_features` directory. You will only need the ImageNet features to replicate our results.



### Clone Repo

Clone (or just download) this reposiroty and replace <code>tasks</code> directory in original Matterport3D simulator with the one in this reposiroty. 

After following the steps above the your file directory should look like this:

```shell
Matterport3D/
	build/					  # should be complied in your machine
	cmake/
	connectivity/		  # store Json connecivity graphs for each scan
	img_features/		  # store precomputed image features, i.e. ResNet-152 features
	include/
	pybind11/         # a dependency of Matterport3D Simulator
	...
	tasks/R2R-judy/	  # replace it with the one in this directory
	...
```





## Usage Instructions





