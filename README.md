# Complete AI Workflow Tutorial (Training + Inferencing)
**This tutorial uses NVIDIA DIGITS for training custom deep neural networks and Nvidia DeepStream SDK for inferencing. I will introduce how to use tools provided by NVIDIA to build your first AI application for object detection.**
**In the end, we will train a dogs detection model and use it to detect any dogs shown in the provided video.**
<p align="center">
	<img src="https://github.com/guoping0408/AI-application/blob/master/Images/result.gif">
</p>



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

Caffe should be configured and built. Now we have to do one last thing for Caffe. First, open the file using gedit:

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
 |___/___\___|___| |_| |___/ 6.1.1
```

DIGITS will store yout training datasets and model snapshots under the `digits/jobs` directory.

To use interactive DIGITS, open your web browser and navigate to `0.0.0.0:5000`.

## USING DIGITS

When you first open the DIGITS home screen by typing `0.0.0.0:5000` on your browser, you would see something like this:
<img src="https://github.com/guoping0408/AI-application/blob/master/Images/digit_home.png">

### Training with DIGITS

Here is the start of our **training phrase**. To train a deep neural network, the most critical thing is data. So, the first thing we do is to tell DIGITS where and what our data is. Since this guide is for object detection, we will have to get the data for object detection. During object detection, the computer not only has to tell what the oject is, but also where that oject is located in the picture; therefore, our data must describe the coordinates or the bounding boxes of the objects in the pictures.

#### Download the Sample Detection Data

There are several object detection data format, for example, KITTI, MS-COCO, and others. Regardless, DIGITS specifically uses KITTI metadata format for ingesting the detection bounding labels. The KITTI metadata format looks like the following:

``` bash
cup 0.00 0 0.00 398.40 249.60 769.60 620.80 0.00 0.00 0.00 0.00 0.00 0.00 0.00
cup 0.00 0 0.00 243.20 288.00 243.20 288.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
cup 0.00 0 0.00 723.20 779.20 723.20 779.20 0.00 0.00 0.00 0.00 0.00 0.00 0.00
```

For the purpose of the tutorial, download and extract sample MS-COCO classes already in DIGITS/KITTI format using the commad at your desired directory:

``` bash
$ wget --no-check-certificate https://nvidia.box.com/shared/static/tdrvaw3fd2cwst2zu2jsi0u43vzk8ecu.gz -O coco.tar.gz

$ tar -xzvf coco.tar.gz
```

The files you have downloaded includes pictures of different groups, which are inside the `images` folder, and their corresponding KITTI meta, which describes their bounding boxes information, inside the `labels` folder.


Now, go back to the DIGITS main page, we select the `Object Detection` option in the drop-down menu on the right hand side. Make sure you are at the **Datasets** page.
<img src="https://github.com/guoping0408/AI-application/blob/master/Images/digit_detect_dataset.png">

After that, type your own username then key in the path of image data and the label data. Below is the sample of the form. Remember to modify the data location to the folder where you extracted the dog dataset:

* Training image folder:  `coco/train/images/dog`
* Training label folder:  `coco/train/labels/dog`
* Validation image folder:  `coco/val/images/dog`
* Validation label folder:  `coco/val/labels/dog`
* Pad image (Width x Height):  `640 x 640`
* Custom classes:  `dontcare, dog`
* Group Name:  `COCO`
* Dataset Name:  `COCO-DOG`
<img src="https://github.com/guoping0408/AI-application/blob/master/Images/path1.png">
<img src="https://github.com/guoping0408/AI-application/blob/master/Images/path2.png">

Then click `Create`. After finished, you have imported your dataset into DIGITS successfully and you are ready to do trainning.

#### Training a Model Using DIGITS
Go back to your DIGITS home page. This time select `Model` tab instead, then select the `Object Detection` option in the drop-down menu on the right hand side.
<img src="https://github.com/guoping0408/AI-application/blob/master/Images/model_page.png">

Make the following settings in the form:

* Select Dataset:  `COCO-DOG`
* Training epochs:  `100`
* Subtract Mean:  `none`
* Solver Type:  `Adam`
* Base learning rate:  `2.5e-05`
* Select `Show advanced learning options`
  * Policy:  `Exponential Decay`
  * Gamma:  `0.99`

<img src="https://github.com/guoping0408/AI-application/blob/master/Images/Model_create1.png">

For the network, select `Custom Network` then copy and paste the content of the [detectnet.prototxt](https://github.com/dusty-nv/jetson-inference/blob/master/data/networks/detectnet.prototxt) from the repo.

In addition, download the pre-trained weights from Googlenet as this will help speed up and stabilize training significantly using the following command at your desired directory:

``` bash
$ wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

```

Finally, fill the path of your downloaded GoogleNet like this

<img src="https://github.com/guoping0408/AI-application/blob/master/Images/Model_create2.png">


# Inferencing—DeepStream

DeepStream SDK brings deep neural networks and other complex processing tasks into a stream processing pipeline for computer vision and intelligent video analytics.

To utilize our models trained on DIGITS, we have to install DeepStream SDK **on our TX2**. First, download the DeepStream SDK version 1.5 from the website [https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads) **onto your TX2. (Notice, we are moving from working on host PC for training deep neural networks to working on TX2 for INFERENCING!)**

Then navigate to the DeepStream package directory and extract the contents into the file system:

``` bash
$ sudo tar xpvf DeepStream_SDK_on_Jetson_1.5_pre-release.tbz2
$ sudo tar xpvf deepstream_sdk_on_jetson .tbz2 -C /
$ sudo tar xpvf deepstream_sdk_on_jetson_models.tbz2 -C /
$ sudo ldconfig
```
DeepStream SDK is installed completely. Now let's test it by running the following command:

``` bash
$ nvgstiva-app -c ${HOME}/configs/PGIE-FP16-CarType-CarMake-CarColor.txt
```

Alternatively, you can run the sample on your own video:

``` bash
$ nvgstiva-app -c ${HOME}/configs/PGIE-FP16-CarType-CarMake-CarColor.txt \
-i /home/nvidia/<path_to_your_own_video>
```

## Using DeepStream to load trained models from DIGITS
Things get exciting when we can use our own models we spent hours training to see the effect! So, let's see how to load our models using DeepDtream.

After downloading the snapshot model and extract it, you should get files similar to these:
<img src="https://github.com/guoping0408/AI-application/blob/master/Images/snapshots.png">

The most important files we need are **.caffemodel** and **deploy.prototxt** files.

Here is the catch. Since the default support network in DeepStream is ResNet, we need to use a custom parsing function for DetectNet that we used in DIGITS for training. Therefore, we have to get the parsing fuction for DetectNet from [https://github.com/AastaNV/DeepStream](https://github.com/AastaNV/DeepStream). Clone the following repo to get the DetectNet parsing function:

``` bash
$ git clone https://github.com/AastaNV/DeepStream
```

The parsing function is in the parser_detectnet folder. Therefore, go to the parser_detectnet folder and build the function:
``` bash
$ cd DeepStream/parser_detectnet
$ make -j8
```

After finished, you will see a **.so** files (e.g. **libnvparsebbox.so**). This is our parsing function. At this point, after we've got our parsing function for DetectNet ready, it's just one step away from integrating the workflow of Deepstream and DIGITS for deep learning!

To utilize our custom parsing function for DectNet, we simply have to modify the config.txt to point to the path to our parsing function so that the **nvgstiva-app** will use custom parsing function instead of the default ResNet one. Here is the sample code of [primary] group in the config.txt:

``` bash
[primary-gie]
enable=1
net-scale-factor=1

# Provide path to our DIGITS model
model-file=file:///home/nvidia/Desktop/DeepStream/Model/DetectNet/snapshot_iter_38600.caffemodel
proto-file=file:///home/nvidia/Desktop/DeepStream/Model/DetectNet/deploy.prototxt
# model-cache=file:///home/nvidia/Desktop/DeepStream/Model/DetectNet/snapshot_iter_38600.caffemodel_b4_fp16.cache
labelfile-path=file:///home/nvidia/Desktop/DeepStream/Model/DetectNet/labels.txt
.
.
.
# Set the parse-func=0 to use the custom parsing function
parse-func=0
parse-bbox-func-name=parse_bbox_custom_detectnet

# Provide the path to our DetectNet parsing function
parse-bbox-lib-name=/home/nvidia/Desktop/DeepStream/parser_detectnet/libnvparsebbox.so
is-classifier=0
output-bbox-name=bboxes
output-blob-names=coverage
```


The above configuration code simply provides a path to our extracted .caffemodel and deploy.prototxt files. In addition, you have to create a label files yourself. In case of this tutorial, create an empty text file and simply write "Dogs" and save it as **labels.txt** and specify its path in the config.txt. It's as simple as the following:

<img src="https://github.com/guoping0408/AI-application/blob/master/Images/label.png">

Also notice that I've comment out the **model-cache=file:** above. After your first run, your TX2 will create a **.cache** file in the same folder as the **.caffemodel** file. The .cache file will store some key information so that when you run the secnod time, we no longer need to wait for a long time while initiating the inference process. (When you uncomment the model-cache=file, remember to replace my .cache file with yours.)


Finally, when we run the nvgstiva-app using the new config.txt, for example,

``` bash
nvgstiva-app -c ~/Desktop/DeepStream/configs/DetectNet.txt -i ~/Downloads/Dogs.mp4
```

We should see that our TX2 successfully detects dogs in our video!

<p align="center">
	<img src="https://github.com/guoping0408/AI-application/blob/master/Images/result.gif">
</p>




