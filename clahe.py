import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("image/image.jpg")
# load the input image from disk and convert it to grayscale
image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=5)
final_img = clahe.apply(image_bw) + 30
cv2.imshow('Original', img)
cv2.imshow('Use Equalization ', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
