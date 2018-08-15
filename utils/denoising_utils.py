import os
import torch

from .common_utils import *
dtype = torch.cuda.FloatTensor


# from https://github.com/DmitryUlyanov/deep-image-prior/blob/master/utils/denoising_utils.py
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.

    Args:
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np




# Function Se implemented as designed in http://www.cs.tut.fi/~foi/papers/Foi-Practical_denoising_clipped-EUSIPCO2008.pdf
coeffs_se = [-3.1389654e-6, -7.0210049e-1, -1.8598057, 3.9881199, -8.3760888, 9.7330082, -6.9596693, 2.9464712, -7.3358565e-1, 9.9630886e-2, -5.7155088e-3]
dtype = torch.cuda.FloatTensor
def p_se(x):
    return sum([(x**p)*coeff for p, coeff in enumerate(coeffs_se)])

def Se(ksi):
    if not torch.is_tensor(ksi):
        ksi = dtype([ksi])
    return 1 - torch.exp(p_se(torch.sqrt(ksi)))

def clipped_var(image, variance):
    return variance*Se(image/variance)*Se((1-image)/variance)# + 0.5*variance*Se(image/variance)*Se((1-image)/variance)

# Coefficients computed by computing polynomial regression on Se against measured method noise
# see http://report.pdf for more information.
def predict_method_noise_std(image, variance):
    image = torch.from_numpy(image).type(dtype)
    pred = torch.sqrt(torch.sum(clipped_var(image, variance)**2)/image.nelement())
    coeffs = [0.567021331273273, 0.42986288350701973, -1.6558506730168887, 1.6735373539304699, -4.5673353213144, 5.847404776912306]
    vars_ = [pred, variance, variance**2, pred*variance, pred**2*variance, pred*variance**2]
    return sum([c * v for c, v in zip(coeffs, vars_)])
