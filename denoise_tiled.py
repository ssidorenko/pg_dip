from datetime import datetime
import os
import sys

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
    interpolate_lr, set_lr, np_to_pil
from utils.tiling_utils import get_regions, image_from_regions

from utils.denoising_utils import get_noisy_image, predict_method_noise_std
from models.skip_network import SkipNetwork


def denoise_region(orig_img_noisy_np, orig_img_np=None, plot=False):
    start_time = datetime.now()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor

    imsize = -1
    sigma = 25
    sigma_ = sigma/255.

    MAX_LEVEL = 5

    orig_img_np = orig_img_noisy_np if orig_img_np is None else orig_img_np
    orig_img_pil = np_to_pil(orig_img_np)
    orig_img_noisy_pil = np_to_pil(orig_img_noisy_np)

    if plot:
        plot_image_grid([orig_img_noisy_np], 4, 5)

    # # Set up parameters and net

    input_depth = 32

    INPUT = 'noise'
    OPT_OVER = 'net'
    KERNEL_TYPE = 'lanczos2'

    tv_weight = 0.0

    OPTIMIZER = 'adam'
    LR = 0.005
    weight_decay = 0.0

    show_every = 100
    RAMPUP_DURATION = 70
    figsize = 3

    num_iter = 1000000000
    reg_noise_std = 1./30. # set to 1./20. for sigma=50
    target_method_noise_std = predict_method_noise_std(orig_img_noisy_np, sigma/255) * 255
    print("Target method noise: {:.4f}".format(target_method_noise_std))
    def get_phase_duration(level, phase):
        if level <= MAX_LEVEL - 1:
            if phase == 'trans':
                return 10+level*4
            elif phase == 'stab':
                return 10+level*3
        else:
            if phase == 'trans':
                return 40
            elif phase == 'stab':
                return 4000

    # In[5]:

    net = SkipNetwork(
        input_channels=input_depth,
        skip_channels=[4]*MAX_LEVEL,
        down_channels=[32]*MAX_LEVEL,
        norm_fun="BatchNorm"
    ).type(dtype)

    mse = torch.nn.MSELoss().type(dtype)

    prev_out = []
    out_avg = None
    last_net = None
    psrn_noisy_last = 0
    psnr_history = []
    overfit_counter = -25
    max_out = None
    prev_psrn_gt_sm = 0.0
    avg_len = 20
    def closure(i, j, max_iter, cur_level, phase, image_target):
        nonlocal out_avg, last_net, psrn_noisy_last, psnr_history, overfit_counter, prev_psrn_gt_sm, max_out
        # note: current_iteration and max_iterations are relative to the current optimize() call
        # however i is not reset to 0 between each optimize() calls

        if reg_noise_std > 0:
            # Adapt regularization noise amplitude to current level.
            # It seems like at lower resolutions, reg. noise is too strong when using values from DIP
            # net_input.data = net_input_saved + (noise.normal_() * reg_noise_std)
            net_input.data = net_input_saved + (noise.normal_() * (reg_noise_std * 10**(-(MAX_LEVEL - cur_level))))

        out = net(net_input)

        if cur_level == MAX_LEVEL:
            prev_out.append(out.detach())
            if len(prev_out) > avg_len:
                del prev_out[0]

        if cur_level == MAX_LEVEL:
            out_avg = sum(prev_out)/len(prev_out)
            psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
            psrn_gt = compare_psnr(orig_img_np, out.detach().cpu().numpy()[0])
            psrn_gt_sm = compare_psnr(orig_img_np, out_avg.detach().cpu().numpy()[0])
            method_noise_mse = np.sqrt(compare_mse(orig_img_noisy_np - out_avg.detach().cpu().type(torch.FloatTensor).numpy()[0], np.zeros(orig_img_np.shape, dtype=np.float32))*255**2)
            if psrn_gt_sm > prev_psrn_gt_sm:
                max_out = out_avg
                prev_psrn_gt_sm = psrn_gt_sm
            if method_noise_mse < target_method_noise_std:
                overfit_counter += 1

            if overfit_counter == 0:
                raise StopIteration()
            psnr_history.append((psrn_gt_sm, method_noise_mse))
        else:
            psnr_history.append((0.0,0.0))

        if plot and (i % show_every == 0 or j == max_iter - 1):
            print("i:{} j:{}/{} phase:{}\n".format(i, j+1, max_iter, phase))
            out_np = var_to_np(out)
            img = np.clip(out_np, 0, 1)
            if cur_level == MAX_LEVEL:
                out_avg = sum(prev_out)/len(prev_out)
            plot_image_grid([img, np.clip(var_to_np(out), 0, 1)], factor=figsize, nrow=2, interpolation=None)
        total_loss = mse(out, image_target)
        total_loss.backward()
        if cur_level == MAX_LEVEL:
            print('Iteration %05d    Loss %f   Noise_stddev %f    PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), method_noise_mse, psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
        else:
            print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
        return out

    i = 0  # Global iteration count, do not reset betwen each phase

    orig_noise = get_noise(
            input_depth,
            INPUT,
            (
                int(orig_img_pil.size[1]),
                int(orig_img_pil.size[0])
            )
        ).type(dtype).detach()
    img_noisy_np = None
    for cur_level in range(1, MAX_LEVEL + 1):
        net.grow()
        net = net.type(dtype)
        s = sum(np.prod(list(p.size())) for p in net.parameters())

        if cur_level != MAX_LEVEL:
            net_input = nn.AvgPool2d(kernel_size=2**(MAX_LEVEL - cur_level))(orig_noise)
        else:
            net_input = orig_noise

        net_input_saved = net_input.data.clone()

        img_noisy_pil = orig_img_noisy_pil.resize(
            (
                orig_img_noisy_pil.size[0] // (2**(MAX_LEVEL - cur_level)),
                orig_img_noisy_pil.size[1] // (2**(MAX_LEVEL - cur_level))
            ),
            Image.ANTIALIAS
        )
        if cur_level != 1:
            prev_img_noisy_np = img_noisy_np

        img_noisy_np = pil_to_np(img_noisy_pil)
        img_noisy_var = np_to_var(img_noisy_np).type(dtype)
        noise = net_input.data.clone()

        # Skip transition and stabilization phases if we're at first level

        for phase in ["trans", "stab"] if cur_level != 1 else ["stab"]:
            # Re-create a new optimizer after each flush()/grow() calls, as we need to let the optimizer know about
            # New or removed model parameters.
            optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)
            if phase == "stab":
                net.flush()
                net = net.type(dtype)
                # print(net)

            for j in range(get_phase_duration(cur_level, phase)):
                # Increase alpha smoothly from 0 at the first iteration to 1 at the last iteration
                alpha = min(j / get_phase_duration(cur_level, phase), 1.0)
                LR_rampup = np.sin((alpha + 1.5) * np.pi)/2 + 0.5
                set_lr(optimizer, LR*LR_rampup)

                if phase == "trans":
                    set_lr(optimizer, LR*LR_rampup)
                    net.update_alpha(alpha)
                    img_noisy_var = np_to_var(interpolate_lr(img_noisy_np, prev_img_noisy_np, alpha)).type(dtype)
                optimizer.zero_grad()
                try:
                    last_out = closure(i, j, get_phase_duration(cur_level, phase), cur_level, phase, img_noisy_var)
                except StopIteration:
                    break

                i += 1
                optimizer.step()
    print("finished, time: {}".format(datetime.now() - start_time))
    out_avg = sum(prev_out)/len(prev_out)
    print("psnr {} dB".format(compare_psnr(orig_img_np, var_to_np(out_avg))))
    return out_avg, psnr_history


def denoise(fname, plot=False, stopping_mode="AMNS"):
    start_time = datetime.now()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    dtype = torch.cuda.FloatTensor
    # dtype = torch.DoubleTensor

    imsize =-1
    plot = False
    sigma = 25
    sigma_ = sigma/255.
    OVERLAP = 16

    patch_size = 128
    patch_stride = 64

    # Add synthetic noise
    orig_img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    orig_img_np = pil_to_np(orig_img_pil)
    np.random.seed(7)
    orig_img_noisy_pil, orig_img_noisy_np = get_noisy_image(orig_img_np, sigma_)

    if plot:
        plot_image_grid([orig_img_np, orig_img_noisy_np], 4, 6);

    regions_n_y = orig_img_np.shape[1]//128
    regions_n_x = orig_img_np.shape[2]//128
    print("Splitting image of shape {} in ({}, {}) regions".format(orig_img_np.shape, regions_n_y, regions_n_x))

    noisy_regions = get_regions(orig_img_noisy_np, regions_n_y, regions_n_x, OVERLAP)
    clean_regions = get_regions(orig_img_np, regions_n_y, regions_n_x, OVERLAP)
    denoised = [[var_to_np(denoise_region(noisy_region, clean_region)[0]) for noisy_region, clean_region in zip(noisy_row, clean_row)] for noisy_row, clean_row in zip(noisy_regions, clean_regions)]
    out = image_from_regions(denoised, OVERLAP)
    print("Patched PSNR: {:.4f}".format(compare_psnr(orig_img_np, out)))
    return out


if __name__ == "__main__":
    stopping_mode = "AMNS"
    if len(sys.argv) > 1:
        stopping_mode = sys.argv[1]

    IMAGES = ["data/denoising/" + image for image in [
        'house.png',
        # 'F16.png',
        # 'lena.png',
        # 'baboon.png',
        # 'kodim03.png',
        # 'kodim01.png',
        # 'peppers.png',
        # 'kodim02.png',
        # 'kodim12.png'
    ]]

    psnrs = []

    for fname in IMAGES:

        img_np = pil_to_np(crop_image(get_image(fname, -1)[0], d=32))

        run1 = denoise(fname, False, stopping_mode)[0]
        run2 = denoise(fname, False, stopping_mode)[0]

        psnr1, psnr2, psnr_avg = [compare_psnr(i, img_np) for i in [run1, run2, 0.5 * (run1 + run2)]]

        print("Run 1: {}\nRun 2: {}\n PSNR of Average: {}".format(psnr1, psnr2, psnr_avg))
        psnrs.append(psnr_avg)

    print("Average PSNR over test set: {}".format(np.mean(psnrs)))
