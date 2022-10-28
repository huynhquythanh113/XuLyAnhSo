import cv2


img = cv2.imread('image/hinhanh.jpg', 0)
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)
th4 = cv2.adaptiveThreshold(img, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
th5 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 71, 2)
th6 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 101, 4)
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'AdaptiveGaussianThresholding 20', 'AdaptiveGaussianThresholding 30', 'AdaptiveGaussianThresholding 40']
images = [img, th1, th2, th3, th4, th5, th6]
xrange = [0, 1, 2, 3, 4, 5, 6]

for i in xrange:
    # plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    # plt.title(titles[i])
    # plt.xticks([]), plt.yticks([])
    cv2.imshow(titles[i], images[i])
cv2.waitKey(0)
cv2.destroyAllWindows()
