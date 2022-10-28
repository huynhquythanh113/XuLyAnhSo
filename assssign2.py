import cv2
import numpy as np
# Load original image
img = cv2.imread('image/image.jpg')
# Create list to store noisy images
images = []
# Generate noisy images using cv2.randn. Can use your own mean and std.
for _ in range(20):
    img1 = img.copy()
    cv2.randn(img1, (0, 0, 0), (50, 50, 50))
    images.append(img+img1)
# For averaging create an empty array, then add images to this array.
img_avg = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.float32)
for im in images:
    img_avg = img_avg+im/20
# Round the float values. Always specify the dtype
img_avg = np.array(np.round(img_avg), dtype=np.uint8)
# Display the images
cv2.imshow('average_image', img_avg)
cv2.imshow('original_image', img)
cv2.imshow('noise_image', images[1])
cv2.waitKey(0)
