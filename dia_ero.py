import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/Resized.png'
img = cv.imread(path, cv.IMREAD_UNCHANGED)
kernel = np.ones((3, 3), np.uint8)
iteration = 1
erosion = cv.erode(img, kernel, iterations=iteration)
dilation = cv.dilate(img, kernel, iterations=iteration)
plt.subplot(1, 3, 1), plt.imshow(img, cmap='Greys_r'),plt.title('ORIGINAL IMAGE')
plt.subplot(1, 3, 2), plt.imshow(erosion, cmap='Greys_r'), plt.title('ERODED IMAGE')
plt.subplot(1, 3, 3), plt.imshow(dilation, cmap='Greys_r'), plt.title('DILATED IMAGE')
# plt.savefig('final_image_name.extension')  # To save figure
plt.show()  # To show figure
'''
cv.imshow('Initial', img)
cv.imshow('Erosion', erosion)
cv.imshow('Dilation', dilation)
'''
cv.waitKey(0)
