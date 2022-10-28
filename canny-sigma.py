from cmath import sqrt
import numpy as np
import argparse
import glob
import cv2


def auto_canny(image, sigma=0.03):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


img = cv2.imread("image/image.jpg")
img1 = auto_canny(img, 0.03)
cv2.imshow('Original', img)
cv2.imshow('Sigma = 0.03 ', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
