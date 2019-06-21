# Complete AI Workflow Tutorial (Training + Inferencing)
**This tutorial uses NVIDIA DIGITS for training custom deep neural networks and Nvidia DeepStream SDK for inferencing. I will introduce how to use tools provided by NVIDIA to build your first AI application for object detection.**


------------


Hardware Prerequisites:

		1. A Host PC with Unbuntu 16.04 or above installed — Training side
		2. NVIDIA RTX 2080 Ti (recommended) — Training side
		3. NVIDIA TX2 — Inferencing side

Software Dependencies:
	(See below instructions. **Be aware of any versions of the toolkit that you install on your devices, since it's critical that you install the right version that fit your development environment.**)
# Training—DIGITS(NVIDIA Deep Learning GPU Training System)

 > **note**: To avoid most dependency issues while using DIGITS and to allow users have more control of their installation environment, I give the somewhat more advanced ways of setting up the DIGITS environments.

## Installing NVIDIA Driver on the Host PC

Assuming you have already used JetPack 4.2 to flash your TX2 or other Jetson, and installed CUDA toolkits on your host PC during the flashing process, you can start installing the driver for your graphic card. It's recommended that you have installed **CUDA 10.0** on your host, since it supports the NVIDIA RTX 2080 Ti graphic card which I use for this tutorial. Other CUDA versions are okay as long as they are compatible with you graphic card.

