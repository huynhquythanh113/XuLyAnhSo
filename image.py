import cv2 as cv
import numpy as np

img = cv.imread("image/img4.jpg", 0)

edges = cv.Canny(img, 100, 200)
cv.imshow("original image", img)
cv.imshow("Canny", edges)

cv.waitKey(0)
cv.destroyAllWindows()
