# Progressively Grown Deep Image Prior

This repository contains the code used for the experiments in my master's thesis. This implementation is based on the original [deep-image-prior](http://github.com/DmitryUlyanov/deep-image-prior) implementation, using concept from [Progressive Growing of GANs](https://arxiv.org/abs/1710.10196) and custom pytorch layers from [pggan-pytorch](https://github.com/nashory/pggan-pytorch/).

The file denoise_pgdip.py runs PG-DIP, optionally with Static or Adaptive Method Noise Stopping, on the color denoising test set.

denoise_mns.py runs the original DIP implementation augmented with Static or Adaptive Method Noise Stopping.

In the folder baseline, is given an executable script denoise_baseline.py to compute baseline Deep
Image Prior results, along with a copy of the Deep Image Prior repository with the following modifications:

-  Add eps=1e-06 to the BatchNorm layer, so that PyTorch avoid using CuDNN for BN as it seems to be a source of numerical instability with some GPUs.
-  Add align_corners=True to the Upsample layer so that it behaves as it used to behave in PyTorch 0.2.
