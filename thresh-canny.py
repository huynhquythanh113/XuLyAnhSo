import cv2
img = cv2.imread("image/image.jpg")  # Read image
t_lower = 0.3
t_upper = 0.3
aperture_size = 5
L2Gradient = True

edge = cv2.Canny(img, t_lower, t_upper,
                 apertureSize=aperture_size,
                 L2gradient=L2Gradient)

cv2.imshow('original', img)
cv2.imshow('double thresh 0.3', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
