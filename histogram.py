import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("image/image.jpg", 0)
histg = cv2.calcHist([img], [0], None, [256], [0, 256])
equ = cv2.equalizeHist(img)

cv2.imshow('Original', img)

cv2.imshow('Use histogram', histg)
cv2.imshow('Use Equalization ', equ)
plt.plot(histg)
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
