import cv2
import numpy as np


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


image = cv2.imread("image/image.jpg")
cv2.imshow('Original image', image)

c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1))

# Specify the data type so that
# float value will be converted to int
log_image = np.array(log_image, dtype=np.uint8)
cv2.imshow('Use log ', log_image)
log_image1 = gammaCorrection(log_image, 2.2)


cv2.imshow('Use log and gamma', log_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
