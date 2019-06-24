# Complete AI Workflow Tutorial (Training + Inferencing)
**This tutorial uses NVIDIA DIGITS for training custom deep neural networks and Nvidia DeepStream SDK for inferencing. I will introduce how to use tools provided by NVIDIA to build your first AI application for object detection.**


------------


### Hardware Prerequisites:

		1. A Host PC with Unbuntu 16.04 or above installed — Training side
		2. NVIDIA RTX 2080 Ti (used in this tutorial) — Training side
		3. NVIDIA TX2 — Inferencing side

### Software Dependencies:

See instructions below. 

(Be aware of any **version of the toolkits** that you will install on your devices, since it's critical that you install the right version which fits your development environment.)

# Flashing your TX2 with Jetpack 4.2

Let's start by setting up your TX2. First go to NVIDIA website to download the latest Jetpack installer [https://developer.nvidia.com/embedded/jetpack](https://developer.nvidia.com/embedded/jetpack) on your host PC. After finish downloading, type the following command to install and launch the Jetpack SDK Manager at the directory where your Jetpack installer is located:

``` bash
$ sudo apt install ./sdkmanager-[version].[build#].deb
$ sdkmanager
```

At this point, you will see a GUI for your Jetpack SDK Manager showing up. Follow the instruction shown on the GUI to complete flashing your TX2 and installing prerequistes softwares on your host PC (or you can see some graphical instruction on the website [https://developer.ridgerun.com/wiki/index.php?title=Installing_JetPack_4.2_-_Nvidia_SDK_Manager](https://developer.ridgerun.com/wiki/index.php?title=Installing_JetPack_4.2_-_Nvidia_SDK_Manager)).

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

### Installing Caffe (a bit complicated)

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
$ cd /usr/lib/x86_64-linux-gnu
***$ sudo ln -s libhdf5_serial.so<version> libhdf5.so***	(Replace the <version> to versions on your computer)
***$ sudo ln -s libhdf5_serial_hl<version> libhdf5_hl.so***	(Replace the <version> to versions on your computer)
```

Then adjust the Makefile.config file to fit your environment. In this tutorial (assuming you have the same environment as mine), we uncomment:

``` bash
# OPENCV_VERSION := 3       ——> OPENCV_VERSION := 3
	(We are using OpenCV 3 because Jetpack 4.2 installed this version as default)
	
# WITH_PYTHON_LAYER := 1    ——> WITH_PYTHON_LAYER := 1
```

And we modify the Makefile.config again by replacing lines "PYTHON_LIB :=" and "INCLUDE_DIRS :=" with the following:

``` bash
PYTHON_LIB := /usr/lib/x86_64-linux-gnu

INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/
```

Replace the CUDA architecture setting in the Makefile.config with the following lines to avoid the "nvcc fatal : Unsupported gpu architecture 'compute_20' issue":

``` bash
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
                                      -gencode arch=compute_35,code=sm_35 \
                                      -gencode arch=compute_50,code=sm_50 \
                                      -gencode arch=compute_52,code=sm_52 \
                                      -gencode arch=compute_60,code=sm_60 \
                                      -gencode arch=compute_61,code=sm_61 \
                                      -gencode arch=compute_61,code=compute_61
```

**Finally, we build Caffe using the following commands:**

``` bash
$ make all -j8
$ make test -j8
$ make runtest -j8
```
(where the '8' in the above commands can be the nubmer of cores in your computer)

Caffe should be configured and built. Now we have to do the last thing for Caffe. First, open the file using gedit:

``` bash
$ gedit ~/.bashrc
```
**Then, add the following two lines inside this file. You can change the paths below to reflect you own**

``` bash
export CAFFE_ROOT=~/caffe		(Here, I cloned caffe to my home directory, replace the paths below to reflect you own)
export PYTHONPATH=~/caffe/python:$PYTHONPATH
```

### Install DIGITS!

Finally, we can start installing DIGTIS on our host computer for training! Let's start by cloning the DIGITS repo from GitHub and get access to some package repositories:

``` bash
$ git clone https://github.com/nvidia/DIGITS

# For Ubuntu 16.04
CUDA_REPO_PKG=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

ML_REPO_PKG=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb

# Install repo packages
wget "$CUDA_REPO_PKG" -O /tmp/cuda-repo.deb && sudo dpkg -i /tmp/cuda-repo.deb && rm -f /tmp/cuda-repo.deb

wget "$ML_REPO_PKG" -O /tmp/ml-repo.deb && sudo dpkg -i /tmp/ml-repo.deb && rm -f /tmp/ml-repo.deb

# Download new list of packages
sudo apt-get update

# Install some dependencies
sudo apt-get install --no-install-recommends git graphviz python-dev python-flask python-flaskext.wtf python-gevent python-h5py python-numpy python-pil python-pip python-scipy python-tk

# Install some python packages
sudo pip install -r $DIGITS_ROOT/requirements.txt
```

### Starting the DIGITS server by the command:

``` bash
$ ./digits-devserver
  ___ ___ ___ ___ _____ ___
 |   \_ _/ __|_ _|_   _/ __|
 | |) | | (_ || |  | | \__ \
 |___/___\___|___| |_| |___/ 6.0-dev
```

DIGITS will store yout training datasets and model snapshots under the `digits/jobs` directory.

To use interactive DIGITS, open your web browser and navigate to `0.0.0.0:5000`.

## USING DIGITS



# Inferencing—DeepStream

DeepStream SDK brings deep neural networks and other complex processing tasks into a stream processing pipeline for computer vision and intelligent video analytics.

To utilize our models trained on DIGITS, we have to install DeepStream SDK **on our TX2**. First, download the DeepStream SDK version 1.5 from the website [https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads) **on your TX2. (Notice, we are moving from working on host PC for training deep neural networks to working on TX2 for INFERENCING!)**


