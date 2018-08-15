"""Original Deep Image Prior implementation for denoising. This code is used to
get the results baseline DIP results in the report. The only changes are the
BatchNorm's epsilon parameter in models.common, and align_corners=True in
Upsample layers."""

from datetime import datetime
import os

import matplotlib.pyplot as plt
from IPython.display import clear_output

import numpy as np
from PIL import Image
from skimage.measure import compare_psnr, compare_mse
import torch
from torch import nn
import torchvision

from utils.common_utils import plot_image_grid, get_noise, \
    np_to_var, var_to_np, pil_to_np, crop_image, get_image, \
    interpolate_lr, set_lr
from utils.denoising_utils import get_noisy_image, predict_method_noise_std
from models.skip_network import SkipNetwork


def denoise(fname, plot=False):
    """Add AWGN with sigma=25 to the given image and denoise it.

    Args:
        fname: Path to the image.
        mode: Stopping mode to use. either "AMNS", "SMNS", or "static".

    Returns:
        A tuple with the denoised image in numpy format as the first element,
        and a history of the PSNR in the second element.

    """
    dtype = torch.cuda.FloatTensor

    sigma = 25
    sigma_ = sigma/255.
    np.random.seed(7)

    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

    if plot:
        plot_image_grid([img_np, img_noisy_np], 4, 6)

    INPUT = 'noise'  # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net'  # 'net,input'

    reg_noise_std = 1./30.  # set to 1./20. for sigma=50
    LR = 0.01
    exp_weight = 0.99  # Exponential averaging coefficient

    OPTIMIZER = 'adam'  # 'LBFGS'
    show_every = 500

    num_iter = 1800
    input_depth = 32
    figsize = 4

    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

    net_input = get_noise(input_depth, INPUT, (img_np.shape[1], img_np.shape[2])).type(dtype).detach()

    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0

    i = 0
    psnr_history = []

    def closure():
        nonlocal i, out_avg, psrn_noisy_last, last_net, psnr_history, ofc, max_out, max_psnr

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        # Smoothing
        if exp_weight is not None:
            if out_avg is None:
                out_avg = out.detach()
            else:
                out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()

        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
        psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])
        psnr_history.append(psrn_gt_sm)

        print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        if plot and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1), np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)

        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5:
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                last_net = [x.data.cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy

        i += 1
        return total_loss

    p = get_params(OPT_OVER, net, net_input)
    try:
        optimize(OPTIMIZER, p, closure, LR, num_iter)
    except StopIteration:
        pass

    return out_avg, psnr_history

if __name__ == "__main__":
    IMAGES = ["data/denoising/" + image for image in [
        'house.png',
        'F16.png',
        'lena.png',
        'baboon.png',
        'kodim03.png',
        'kodim01.png',
        'peppers.png',
        'kodim02.png',
        'kodim12.png'
    ]]

    psnrs = []

    for fname in IMAGES:

        img_np = pil_to_np(crop_image(get_image(fname, -1)[0], d=32))

        run1 = var_to_np(denoise(fname, plot, stopping_mode)[0])
        run2 = var_to_np(denoise(fname, plot, stopping_mode)[0])

        psnr1, psnr2, psnr_avg = [compare_psnr(i, img_np) for i in [run1, run2, 0.5 * (run1 + run2)]]

        print("Run 1: {}\nRun 2: {}\n PSNR of Average: {}".format(psnr1, psnr2, psnr_avg))
        psnrs.append(psnr_avg)

    print("Average PSNR over test set: {}".format(np.mean(psnrs)))
