from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os
import scipy.misc as sm
import skimage
from skimage.feature import canny
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
import cv2
from scipy import ndimage

from scipy import misc
import numpy as np


def visualize(imgs, format=None):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        plt_idx = i+1
        plt.subplot(4, 2, plt_idx)
        plt.imshow(img, cmap=cm.gray, vmin=0, vmax=255)
    plt.show()


def gaussian_kernel(size, sigma=2):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    #pi_4 = np.pi / 4
    #pi_2 = np.pi / 2
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                # theta = D[i,j] #* 180 / np.pi #angle in degrees
                #theta_mod = theta % np.pi
                q = 255
                r = 255
                #alpha = None

               # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                """
                if (0 <= theta_mod < pi_4):
                    alpha = np.abs(np.tan(theta_mod))
                    q = (alpha * img[i + 1, j + 1]) + ((1 - alpha) * img[i, j + 1])
                    r = (alpha * img[i - 1, j - 1]) + ((1 - alpha) * img[i, j - 1]) 
                    
                elif (pi_4 <= theta_mod < pi_2):
                    alpha = np.abs(1./np.tan(theta_mod))
                    q = (alpha * img[i + 1, j + 1]) + ((1 - alpha) * img[i + 1, j])
                    r = (alpha * img[i - 1, j - 1]) + ((1 - alpha) * img[i - 1, j])
                    
                elif (pi_2 <= theta_mod < (3*pi_4)):
                    alpha = np.abs(1./np.tan(theta_mod))
                    q = (alpha * img[i + 1, j - 1]) + ((1 - alpha) * img[i + 1, j])
                    r = (alpha * img[i - 1, j + 1]) + ((1 - alpha) * img[i - 1, j])
                
                elif ((3*pi_4) <= theta_mod < np.pi):
                    alpha = np.abs(np.tan(theta_mod))
                    q = (alpha * img[i + 1, j - 1]) + ((1 - alpha) * img[i, j - 1])
                    r = (alpha * img[i - 1, j + 1]) + ((1 - alpha) * img[i, j + 1])
                """
                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def hysteresis(img, weak, strong=255):

    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img


def Canny_detector(img):
    """ Your implementation instead of skimage """
    img = np.zeros(shape=(130, 130), dtype=np.float32)

    img_filtered = convolve(img, gaussian_kernel(5, sigma=1.8))
    grad, theta = sobel_filters(img_filtered)
    img_nms = non_max_suppression(grad, theta)
    img_thresh, weak, strong = threshold(
        img_nms, lowThresholdRatio=0.07, highThresholdRatio=0.19)
    img_final = hysteresis(img_thresh, weak, strong=strong)

    return img_final


img = cv2.imread("image/img.jpg")

canny_img = Canny_detector(img)
plt.figure(figsize=(20, 40))
plt.imshow(canny_img, cmap='gray', vmin=0, vmax=100)
plt.show()
