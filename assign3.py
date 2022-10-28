import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import math


def plot_input(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()


def zero_cross_detection(image):
    z_c_image = np.zeros(image.shape)

    for i in range(0, image.shape[0]-1):
        for j in range(0, image.shape[1]-1):
            if image[i][j] > 0:
                if image[i+1][j] < 0 or image[i+1][j+1] < 0 or image[i][j+1] < 0:
                    z_c_image[i, j] = 1
            elif image[i][j] < 0:
                if image[i+1][j] > 0 or image[i+1][j+1] > 0 or image[i][j+1] > 0:
                    z_c_image[i, j] = 1
    return z_c_image


def handle_img_padding(img1, img2):
    M1, N1 = img1.shape[:2]
    M2, N2 = img2.shape[:2]
    padding_x = np.abs(M2 - M1)/2
    padding_y = np.abs(N2 - N1)/2
    img2 = img2[int(padding_x):int(M1+padding_x),
                int(padding_y): N1+int(padding_y)]
    return img2


LoG_kernel = np.array([
    [0, 0,  1, 0, 0],
    [0, 1,  2, 1, 0],
    [1, 2, -16, 2, 1],
    [0, 1,  2, 1, 0],
    [0, 0,  1, 0, 0]
])
original_img = cv2.imread("image/image.jpg")

plt.imshow(original_img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()
log_img = convolve2d(original_img, LoG_kernel)
plot_input(log_img, 'LoG Image')
plt.imshow(log_img, cmap='gray')
plt.title('LoG Image'), plt.xticks([]), plt.yticks([])
plt.show()
zero_crossing_log = zero_cross_detection(log_img)
zero_crossing_log = handle_img_padding(original_img, zero_crossing_log)
plt.imshow(zero_crossing_log, cmap='gray')
plt.title('Zero Crossing-LoG'), plt.xticks([]), plt.yticks([])
plt.show()
