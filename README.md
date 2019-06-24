# Complete AI Workflow Tutorial (Training + Inferencing)
**This tutorial uses NVIDIA DIGITS for training custom deep neural networks and Nvidia DeepStream SDK for inferencing. I will introduce how to use tools provided by NVIDIA to build your first AI application for object detection.**


------------


### Hardware Prerequisites:

		1. A Host PC with Unbuntu 16.04 or above installed — Training side
		2. NVIDIA RTX 2080 Ti (used in this tutorial) — Training side
		3. NVIDIA TX2 — Inferencing side

### Software Dependencies:

See below instructions. 

(Be aware of any **version of the toolkits** that you will install on your devices, since it's critical that you install the right version that fit your development environment.)

# Training—DIGITS(NVIDIA Deep Learning GPU Training System)

 > **note**: To avoid most dependency issues while using DIGITS and to allow users have more control of their installation environment, I give the somewhat more advanced ways of setting up the DIGITS environments natively.
 
### Installing some software prerequisites on the host PC before installing DIGITS:
		1. Nvidia driver
		2. CUDA 10.0
		3. cuDNN
		4. Caffe
		
### Installing NVIDIA Driver on the Host PC

Assuming you have already used JetPack 4.2 to flash your TX2 or other Jetson, and installed CUDA toolkits on your host PC during the flashing process, you can start installing the driver for your graphic card. It's recommended that you have installed **CUDA 10.0** on your host, since it supports the NVIDIA RTX 2080 Ti graphic card which I use for this tutorial. Other CUDA versions are okay as long as they are compatible with you graphic card.

First, run the below command to search for current nvidia driver version available:

``` bash
$ apt-cache search nvidia-driver
```

And, install the newest version of nvidia driver version available:

``` bash
$ sudo apt-get install nvidia-driver-418
$ sudo reboot
```

After rebooting, it should shows something similar on the command line if your NVIDIA driver is installed successfully:

``` bash
$ nvidia-smi
Fri Jun 21 17:44:33 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  Off  | 00000000:01:00.0  On |                  N/A |
| 41%   33C    P8     6W / 260W |    625MiB / 10986MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+
```
### Verifying CUDA 10.0

To check your CUDA version:

``` bash
$ nvcc --version
```

To verify the CUDA toolkit and NVIDIA driver are working, run some tests that come with the CUDA samples:

``` bash
$ cd /usr/local/cuda/samples
$ sudo make
$ cd bin/x86_64/linux/release/
$ ./deviceQuery
$ ./bandwidthTest --memory=pinned
```

### Installing cuDNN

Download the cuDNN runtime library and cuDNN developer library that fit your environment. In this tutorial, we pick the one for CUDA 10.0 and Ubuntu 16.04 from the webpage below

[`https://developer.nvidia.com/rdp/cudnn-download`](https://developer.nvidia.com/rdp/cudnn-download)
	
Then install the packages with the following commands:

``` bash
$ sudo dpkg -i libcudnn<version>_amd64.deb
$ sudo dpkg -i libcudnn-dev_<version>_amd64.deb
```

### Installing Caffe

Clone the caffe-0.15 branch to your desired directory by the following command:

``` bash
$ git clone -b caffe-0.15 https://github.com/NVIDIA/caffe
```

Then **build** the caffe using the following instructions:

First, run the commands:
``` bash
$ cp Makefile.config.example Makefile.config
```

And, install the following packages to meet the prerequistes for installing caffe:

``` bash
$ sudo apt-get install git
$ sudo apt-get install libprotobuf-dev libleveldb-dev libopencv-dev libsnappy-dev
$ sudo apt-get install libhdf5-serial-dev protobuf-compiler
$ sudo apt-get install --no-install-recommends libboost-all-dev
$ sudo apt-get install libatlas-base-dev
$ sudo apt-get install python-dev
$ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```

Then adjust the Makefile.config file to fit your environment. In this tutorial (assuming you have the same environment as mine), we uncomment:

``` bash
# OPENCV_VERSION := 3  ——> OPENCV_VERSION := 3
```

Replace the CUDA architecture setting by the following to avoid the "nvcc fatal : Unsupported gpu architecture 'compute_20' issue":

``` bash
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
                                      -gencode arch=compute_35,code=sm_35 \
                                      -gencode arch=compute_50,code=sm_50 \
                                      -gencode arch=compute_52,code=sm_52 \
                                      -gencode arch=compute_60,code=sm_60 \
                                      -gencode arch=compute_61,code=sm_61 \
                                      -gencode arch=compute_61,code=compute_61
```
