# Progressively Grown Deep Image Prior (PG-DIP)

This repository contains the code used for the experiments in my master's thesis. This implementation is based on the original [deep-image-prior](http://github.com/DmitryUlyanov/deep-image-prior) implementation, using concept from [Progressive Growing of GANs](https://arxiv.org/abs/1710.10196) and some custom PyTorch layers from [pggan-pytorch](https://github.com/nashory/pggan-pytorch/).

This code has been tested with PyTorch 0.4 and a GTX 1080 Ti.

## Abstract 

Focusing on the task of color image denoising, this work improves Deep Image Prior (DIP) in terms of denoising performance, speed, and scalability. DIP is an image-restoration method based on Convolutional Neural Networks with the particularity that it does not use training data. First, we analyze the limitations of DIP in the denoising task and we improve its denoising ability by developing a task-specific regularization method, substantially improving on the baseline and reducing the gap towards benchmarks. Then, we propose an adaptation of the Progressively Grown Generative Adversarial Networks to DIP, yielding a 1.6x speed-up over the original implementation and im- proving denoising performance as well. Last, we propose a tiling method so that the algorithm can be scaled to arbitrarily large images.

## Notes

The file `denoise_pgdip.py` runs PG-DIP, optionally with Static or Adaptive Method Noise Stopping, on the color denoising test set.

`denoise_mns.py` runs the original DIP implementation augmented with Static or Adaptive Method Noise Stopping.

In the folder `baseline`, is given an executable script `denoise_baseline.py` to compute baseline Deep
Image Prior results, along with a copy of the Deep Image Prior repository with the following modifications:

-  Add eps=1e-06 to the BatchNorm layer, so that PyTorch avoid using CuDNN for BN as it seems to be a source of numerical instability with some GPUs.
-  Add align_corners=True to the Upsample layer so that it behaves as it used to behave in PyTorch 0.2.
