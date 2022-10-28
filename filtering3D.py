# Python program to explain cv2.blur() method

# importing cv2
import cv2
from skimage.util import random_noise

# Reading an image in default mode
image = cv2.imread("image/img7.jpg")

# Window name in which image is displayed
window_name = 'Image original'

# ksize
ksize = (30, 30)
image1 = cv2.blur(image, ksize, cv2.BORDER_DEFAULT)
img_b_blur_3 = cv2.boxFilter(image, -1, (3, 3), normalize=False)
median = cv2.medianBlur(image, 5)
# Displaying the image
cv2.imshow("After medianBlur", image1)
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
