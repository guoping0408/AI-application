# Complete AI Workflow Tutorial (Training + Inferencing)
**This tutorial uses NVIDIA DIGITS for training custom deep neural networks and Nvidia DeepStream SDK for inferencing. I will introduce how to use tools provided by NVIDIA to build your first AI application for object detection.**


------------


### Hardware Prerequisites:

		1. A Host PC with Unbuntu 16.04 or above installed — Training side
		2. NVIDIA RTX 2080 Ti (recommended) — Training side
		3. NVIDIA TX2 — Inferencing side

### Software Dependencies:

See below instructions. 

(Be aware of any **version of the toolkits** that you will install on your devices, since it's critical that you install the right version that fit your development environment.)

# Training—DIGITS(NVIDIA Deep Learning GPU Training System)

 > **note**: To avoid most dependency issues while using DIGITS and to allow users have more control of their installation environment, I give the somewhat more advanced ways of setting up the DIGITS environments natively.

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
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1042      G   /usr/lib/xorg/Xorg                           247MiB |
|    0      1475      G   compiz                                        91MiB |
|    0      5153      G   ...uest-channel-token=11217275582148305812   129MiB |
|    0     28577      C   python2                                      153MiB |
+-----------------------------------------------------------------------------+
```


