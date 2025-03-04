import numpy as np
import math
from scipy.optimize import minimize
from tqdm import tqdm
# from main import generate_full_image
from utils import generateFullRef
from scipy.ndimage import convolve, gaussian_filter
import matplotlib.pyplot as plt
from scipy import fft

def FFT(p,q):
    SIZE = p.shape
    x_grid, y_grid = np.meshgrid(np.arange(SIZE[0]), np.arange(SIZE[1]))
    wx = (2 * np.pi * x_grid) / SIZE[0]
    wy = (2 * np.pi * y_grid) / SIZE[1]
    Cp = fft.fft2(p)
    Cq = fft.fft2(q)
    C = -1j * (wx * Cp + wy * Cq) / (wx**2 + wy**2 + 1e-100)
    Z = np.abs(fft.ifft2(C))
    p = np.real(fft.ifft2(1j * wx * C))
    q = np.real(fft.ifft2(1j * wy * C))
    return Z,p,q

def gaussian(I, sigma=1):
    return gaussian_filter(I, sigma=sigma)

def laplacian(I):
    kernel = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    return convolve(I, kernel, mode='constant', cval=0)

def smoothing(smoothing_func,p,q,Z):
    if smoothing_func == FFT:
        Z,p,q = FFT(p,q)
    else:
        p = gaussian(p)
        q = gaussian(q)
        Z = gaussian(Z) + p + q
    return Z, p, q


def recoverdepth(image, s, alpha, lambda_param, max_iter, smoothingFunc, size=[64, 64]):
    # p = np.zeros((size[0], size[1]))
    # q = np.zeros((size[0], size[1]))
    # Z = np.zeros((size[0], size[1]))

    # p = np.random.normal(0, 1, (size[0], size[1]))
    # q = np.random.normal(0, 1, (size[0], size[1]))
    # Z = np.random.normal(0, 1, (size[0], size[1]))

    p = np.ones((size[0], size[1]))
    q = np.ones((size[0], size[1]))
    Z = np.ones((size[0], size[1]))

    E = image
    E_mask = image.copy()
    E_mask[image>0] = 1
    E_mask[image<=0] = 0

    x_grid, y_grid = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    kernel = [[0,0.25,0],[0.25,0,0.25],[0,0.25,0]]

    for k in tqdm(range(max_iter)):
        p_bar = convolve(p, kernel, mode='constant', cval=0)
        q_bar = convolve(q, kernel, mode='constant', cval=0)
        R =  E_mask * ((-s[0] * p - s[1] * q + s[2]) / np.sqrt(1 + p**2 + q**2)) ** alpha
        s_norm = np.sqrt(1 + s[0]**2 + s[1]**2)
        dR_dp = alpha*(R**(alpha-1))*((- s[0] / np.sqrt(1 + p**2 + q**2)) + ((-s[0]  * p - s[1] * q + s[2]) * (-p / ((1 + p**2 + q**2) ** (3 / 2)))))/s_norm
        dR_dq = alpha*(R**(alpha-1))*((- s[1] / np.sqrt(1 + p**2 + q**2)) + ((-s[0]  * p - s[1] * q + s[2]) * (-q / ((1 + p**2 + q**2) ** (3 / 2)))))/s_norm
        p = p_bar + (1 / (4 * lambda_param)) * (E - R) * dR_dp
        q = q_bar + (1 / (4 * lambda_param)) * (E - R) * dR_dq
        # print(p, p.min(), p.max())
        # quit()
        Z, p, q = smoothing(smoothingFunc,p,q,Z)
    return p, q, Z