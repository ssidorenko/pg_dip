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


def denoise(fname, plot=True, stopping_mode="AMNS"):
    """Add AWGN with sigma=25 to the given image and denoise it.

    Args:
        fname: Path to the image.
        mode: Stopping mode to use. either "AMNS", "SMNS", or "static".

    Returns:
        A tuple with the denoised image in numpy format as the first element,
        and a history of the PSNR in the second element.

    """
    start_time = datetime.now()
    dtype = torch.cuda.FloatTensor

    imsize = -1
    sigma = 25
    sigma_ = sigma/255.

    MAX_LEVEL = 5

    # Fix the random seed to that the noisy image are identical everytime.
    # This is mandatory as in the report we compute PSNR by averaging two runs.
    # If the runs used images corrupted with different noise, the noise would
    # cancel out, artificially inflating PSNR.
    np.random.seed(7)
    orig_img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    orig_img_np = pil_to_np(orig_img_pil)

    # Comment the following line and uncomment the next when ground truth is not known
    orig_img_noisy_pil, orig_img_noisy_np = get_noisy_image(orig_img_np, sigma_)
    # orig_img_noisy_pil, orig_img_noisy_np = orig_img_pil, orig_img_np

    if plot:
        plot_image_grid([orig_img_noisy_np], 4, 5)

    # # Set up parameters and net

    input_depth = 3 if "snail" in fname else 32

    INPUT = 'noise'
    OPT_OVER = 'net'
    KERNEL_TYPE = 'lanczos2'

    tv_weight = 0.0

    OPTIMIZER = 'adam'
    LR = 0.01
    weight_decay = 0.0

    show_every = 100
    RAMPUP_DURATION = 70
    figsize = 10

    def get_reg_noise_std(sigma):
        return sigma/(60*25)+1/60

    reg_noise_std = get_reg_noise_std(sigma)
    target_method_noise_std = predict_method_noise_std(orig_img_noisy_np, sigma/255) * 255
    print("Target method noise: {:.4f}".format(target_method_noise_std))


    def get_phase_duration(level, phase):
        if level <= MAX_LEVEL - 1:
            if phase == 'trans':
                return 70
            elif phase == 'stab':
                return 50
        else:
            if phase == 'trans':
                return 100
            elif phase == 'stab':
                return 2000 # tmp for computing method noise


    net = SkipNetwork(
        input_channels=input_depth,
        skip_channels=[4, 4, 4, 4, 4],
        down_channels=[128, 128, 128, 128, 128],
        norm_fun="BatchNorm"
    ).type(dtype)

    exp_weight = 0.99

    mse = torch.nn.MSELoss().type(dtype)

    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print('Number of params for D: %d' % s)

    last_out = None
    out_avg = None
    psrn_noisy_last = 0
    psnr_history = []
    overfit_counter = -25

    def closure(i, j, max_iter, cur_level, phase, image_target):
        """Innermost loop of the optimization procedure."""
        nonlocal out_avg, last_net, psrn_noisy_last, psnr_history, overfit_counter

        # note: j and max_iter are relative to the current phase
        # i counts the total number of iterations since the beginning of execution

        if reg_noise_std > 0:
            # Adapt regularization noise amplitude to current level.
            # It seems like at lower resolutions, reg. noise is too strong when using values from DIP
            net_input.data = net_input_saved + (noise.normal_() * (reg_noise_std * 10**(-(MAX_LEVEL - cur_level))))

        out = net(net_input)

        # If at last level, start computing exponential moving average
        if exp_weight is not None and cur_level == MAX_LEVEL:
            if out_avg is None:
                out_avg = out.detach()
            else:
                out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        if cur_level == MAX_LEVEL:
            # Measure PSNR
            psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
            psrn_gt = compare_psnr(orig_img_np, out.detach().cpu().numpy()[0])
            psrn_gt_avg = compare_psnr(orig_img_np, out_avg.detach().cpu().numpy()[0])
            method_noise_mse = np.sqrt(compare_mse(orig_img_noisy_np - out_avg.detach().cpu().type(torch.FloatTensor).numpy()[0], np.zeros(orig_img_np.shape, dtype=np.float32))*255**2)

            if method_noise_mse < target_method_noise_std:
                overfit_counter += 1

            if overfit_counter == 0:
                raise StopIteration()
            psnr_history.append((psrn_gt_avg, method_noise_mse))

        if plot and (i % show_every == 0 or j == max_iter - 1):
            print("i:{} j:{}/{} phase:{}\n".format(i, j+1, max_iter, phase))
            out_np = var_to_np(out)
            img = np.clip(out_np, 0, 1)
            plot_image_grid([img], factor=figsize, nrow=2)

        total_loss = mse(out, image_target)
        total_loss.backward()

        if cur_level == MAX_LEVEL:
            print('Iteration %05d    Loss %f   Noise_stddev %f    PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), method_noise_mse, psrn_noisy, psrn_gt, psrn_gt_avg), '\r', end='')
        else:
            print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
        if i%100 == 0:
            print("")

        return total_loss, out

    i = 0  # Global iteration count

    # Init fixed random code vector.
    orig_noise = get_noise(
            input_depth,
            INPUT,
            (
                int(orig_img_pil.size[1]),
                int(orig_img_pil.size[0])
            )
        ).type(dtype).detach()
    img_noisy_np = None

    # Iterate over each resolution level
    for cur_level in range(1, MAX_LEVEL + 1):
        net.grow()
        net = net.type(dtype)
        print("Increased network size")
        s = sum(np.prod(list(p.size())) for p in net.parameters())
        print('Number of params: %d' % s)

        # Downsample z
        if cur_level != MAX_LEVEL:
            net_input = nn.AvgPool2d(kernel_size=2**(MAX_LEVEL - cur_level))(orig_noise)
        else:
            net_input = orig_noise

        # Save a copy of z as z will be perturbed by normal noise at each iteration
        net_input_saved = net_input.data.clone()

        img_noisy_pil = orig_img_noisy_pil.resize(
            (
                orig_img_noisy_pil.size[0] // (2**(MAX_LEVEL - cur_level)),
                orig_img_noisy_pil.size[1] // (2**(MAX_LEVEL - cur_level))
            ),
            Image.ANTIALIAS
        )

        # Save the downsampled noisy image from the previous level for when we need to interpolate it
        # with the current resolution

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

            # Flush the network before starting the stabilization phase
            if phase == "stab":
                net.flush()
                net = net.type(dtype)

            for j in range(get_phase_duration(cur_level, phase)):
                # Increase alpha smoothly from 0 at the first iteration to 1 at the last iteration
                alpha = min(j / RAMPUP_DURATION, 1.0)
                LR_rampup = np.sin((alpha + 1.5) * np.pi)/2 + 0.5
                set_lr(optimizer, LR*LR_rampup)

                if phase == "trans":
                    set_lr(optimizer, LR*LR_rampup)
                    net.update_alpha(alpha)
                    img_noisy_var = np_to_var(interpolate_lr(img_noisy_np, prev_img_noisy_np, alpha)).type(dtype)

                optimizer.zero_grad()
                try:
                    _, last_out = closure(i, j, get_phase_duration(cur_level, phase), cur_level, phase, img_noisy_var)
                except StopIteration:
                    break
                i += 1
                optimizer.step()
    print("finished, time: {}".format(datetime.now() - start_time))
    return out_avg, psnr_history

if __name__ == "__main__":
    stopping_mode = "AMNS"
    if len(sys.argv) > 1:
        stopping_mode = sys.argv[1]

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

    for fname in self.IMAGES:

        img_np = pil_to_np(crop_image(get_image(fname, -1)[0], d=32))

        run1, _  = var_to_np(denoise(fname, plot, stopping_mode)[0])
        run2, _  = var_to_np(denoise(fname, plot, stopping_mode)[0])

        psnr1, psnr2, psnr_avg = [compare_psnr(i, img_np) for i in [run1, run2, 0.5 * (run1 + run2)]]

        print("Run 1: {}\nRun 2: {}\n PSNR of Average: {}".format(psnr1, psnr2, psnr_avg))
        psnrs.append(psnr_avg)

    print("Average PSNR over test set: {}".format(np.mean(psnrs)))
