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

``` bash
$ sudo apt-get install -y apt-transport-https curl
$ cat <<EOF | sudo tee /etc/apt/sources.list.d/cuda.list > /dev/null
deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /
EOF
$ curl -s \
 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub \
 | sudo apt-key add -
$ cat <<EOF | sudo tee /etc/apt/preferences.d/cuda > /dev/null
Package: *
Pin: origin developer.download.nvidia.com
Pin-Priority: 600
EOF
$ sudo apt-get update && sudo apt-get install -y --no-install-recommends cuda-drivers
$ sudo reboot
```
