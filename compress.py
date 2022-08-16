# Import functions and libraries
import time
import numpy as np
import sys
import scipy
import cv2
from numpy import r_
from scipy import fftpack
from PIL import Image
import os

def do_compression(img_path, filename):
    print(img_path)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=sys.maxsize)

    # get img_name
    print(filename)

    start = time.time()

    GRID_SIZE = 8

    # Reading Image
    im = cv2.imread(img_path, 1)

    if im.shape[0] * im.shape[1] < 2073600: # 1920 * 1080
        GRID_SIZE = 4
    if im.shape[0] * im.shape[1] < 10000: # 100 * 100
        GRID_SIZE = 2

    # Splitting into Red, Green, Blue channel
    b, g, r = cv2.split(im)


    def dct2(a):
        x = scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')
        return x


    imsize = r.shape
    dct_r = np.zeros(imsize)
    dct_g = np.zeros(imsize)
    dct_b = np.zeros(imsize)

    # 8 X 8 dct of 64 pixels.
    for i in r_[:imsize[0]:GRID_SIZE]:
        for j in r_[:imsize[1]:GRID_SIZE]:
            dct_r[i:(i + GRID_SIZE), j:(j + GRID_SIZE)] = dct2(r[i:(i + GRID_SIZE), j:(j + GRID_SIZE)])
            dct_g[i:(i + GRID_SIZE), j:(j + GRID_SIZE)] = dct2(g[i:(i + GRID_SIZE), j:(j + GRID_SIZE)])
            dct_b[i:(i + GRID_SIZE), j:(j + GRID_SIZE)] = dct2(b[i:(i + GRID_SIZE), j:(j + GRID_SIZE)])

    thresh = 0.01
    if im.shape[0] * im.shape[1] > 2073600: # 1920 * 1080
        thresh = 0.02

    # Removing numbers less than threshold * max(color plane)
    def thresholding(x):
        x = x * (abs(x) > (thresh * np.max(x)))
        n_zeros = x.shape[0] * x.shape[1] - np.count_nonzero(x)
        print("N_zeros:", n_zeros)
        print("thres : ", thresh)
        return x, n_zeros


    # Thresholding of channels
    dct_r, dct_r_n_zeros = thresholding(dct_r)
    dct_b, dct_b_n_zeros = thresholding(dct_b)
    dct_g, dct_g_n_zeros = thresholding(dct_g)

    extent_of_compression = ((dct_r_n_zeros + dct_b_n_zeros + dct_g_n_zeros) / 3) / (
                dct_r.shape[0] * dct_r.shape[1])  ## This can be used to further adjust the threshold
    extent_of_compression = extent_of_compression * 100;
    print("Extent of compression: ", extent_of_compression)
    print("GRID_SIZE : ", GRID_SIZE)
    end = time.time()
    print("TOTAL RUNNING TIME: ", end - start)

    # now implement pillow on extent of compression

    pillow_var = 3;
    pillow_compression_quality = 100 - extent_of_compression + pillow_var;
    pillow_compression_quality = int(pillow_compression_quality)
    image = Image.open(img_path);
    output_path = "output"
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(f"{output_path}/compressed.jpg", quality = pillow_compression_quality)
    return extent_of_compression;